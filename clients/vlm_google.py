"""
Google VLM client: Gemini 3.1 Pro constrained to 14 labels.

Sends the full image to Gemini 3.1 Pro with a prompt that explicitly lists
all 14 class names and instructs the model to count precisely.
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict

from google import genai
from google.genai import types

from config import CLASSES

MODEL = "gemini-3.1-pro-preview"

FROZEN_PROMPT = f"""You are an inventory counting assistant. Examine the image and count every
visible item that matches one of the following 14 classes:

{', '.join(CLASSES)}

Rules:
- ONLY use the exact class names listed above (case-sensitive).
- Count each individual item you can see. If there are 3 apples, report "apple": 3.
- If a class is not present, do NOT include it.
- Do NOT invent classes outside the list.
- Respond ONLY with valid JSON in this exact format (no markdown, no explanation):

{{"inventory": {{"class_name": count, ...}}}}

Example: {{"inventory": {{"apple": 3, "banana": 1, "tomato": 2}}}}
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def _load_image(image_path: str) -> types.Part:
    """Load image as a Part for the Gemini API."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    data = path.read_bytes()
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def identify(image_path: str) -> Dict[str, int]:
    """Send image to Gemini 3.1 Pro and return inventory dict."""
    client = _get_client()
    image_part = _load_image(image_path)

    response = client.models.generate_content(
        model=MODEL,
        contents=[FROZEN_PROMPT, image_part],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )

    raw = response.text.strip()
    return _parse_response(raw)


def _parse_response(raw: str) -> Dict[str, int]:
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    raw_inventory = data.get("inventory", data)
    if not isinstance(raw_inventory, dict):
        return {}

    valid_classes = set(CLASSES)
    inventory: Dict[str, int] = {}
    for key, value in raw_inventory.items():
        key_clean = key.strip().lower()
        if key_clean in valid_classes:
            try:
                count = int(value)
                if count > 0:
                    inventory[key_clean] = count
            except (ValueError, TypeError):
                continue
    return inventory
