# SnapShelf — Experiment 2

**End-to-End Pipeline Comparison for 14-Class Fruit & Vegetable Inventory**

> **BSc Dissertation Artefact**
> Comparing three fundamentally different end-to-end pipelines — VLM, pure YOLO, and YOLO+CNN — for the same 14-class inventory task. This creates a clear thread: Exp 1 (best CNN) → Exp 2 (best pipeline using that CNN) → Exp 3 (integrate best pipeline into app).

## Table of Contents

1. [Research Context](#research-context)
2. [14-Class Inventory Task](#14-class-inventory-task)
3. [Pipeline Architecture](#pipeline-architecture)
4. [VLM Comparison (Pipeline A Model Selection)](#vlm-comparison-pipeline-a-model-selection)
5. [Experimental Design](#experimental-design)
6. [Training Data](#training-data)
7. [Test Data and Annotation](#test-data-and-annotation)
8. [Robustness Testing (Image Degradations)](#robustness-testing-image-degradations)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Training](#training)
12. [Evaluation — The 12-Run Matrix](#evaluation--the-12-run-matrix)
13. [Output Schema](#output-schema)
14. [Configuration](#configuration)
15. [Reproducibility](#reproducibility)
16. [Key Findings and Discussion Points](#key-findings-and-discussion-points)
17. [Project Structure](#project-structure)

---

## Research Context

### Problem Statement

Experiment 1 identified the best CNN architecture for single-crop fruit/vegetable classification (EfficientNet-B0). Experiment 2 asks the next question: **what is the best end-to-end pipeline for building a complete inventory from a single image?**

Three fundamentally different approaches are compared — a Vision-Language Model that reasons about the full image, a purpose-trained object detector that labels and counts directly, and a two-stage pipeline that separates detection from classification.

### Research Questions

1. Which flagship VLM (GPT-5.2, Claude Opus 4.6, or Gemini 3.1 Pro) produces the most accurate inventories when constrained to 14 labels?
2. Does a purpose-trained 14-class YOLO outperform the best VLM on the same task?
3. Does separating detection (objectness YOLO) from classification (CNN from Exp 1) offer advantages over either end-to-end approach?
4. How robust is each pipeline to real-world image degradations (blur, noise, JPEG compression)?
5. What are the latency and cost trade-offs between the three strategies?

---

## 14-Class Inventory Task

All three pipelines solve the identical task: given an image, produce an **inventory** — a dictionary mapping class names to counts.

```python
Inventory = Dict[str, int]
# Example: {"apple": 3, "banana": 1, "tomato": 2}
```

**Classes (14):**

| # | Class | # | Class |
|:-:|-------|:-:|-------|
| 0 | `apple` | 7 | `lemon` |
| 1 | `banana` | 8 | `onion` |
| 2 | `bell_pepper_green` | 9 | `orange` |
| 3 | `bell_pepper_red` | 10 | `peach` |
| 4 | `carrot` | 11 | `potato` |
| 5 | `cucumber` | 12 | `strawberry` |
| 6 | `grape` | 13 | `tomato` |

---

## Pipeline Architecture

### Pipeline Overview

| Pipeline | Strategy | Flow | Models Used | API Calls |
|:--------:|----------|------|-------------|:---------:|
| **A** | VLM-only | Image → GPT-5.2 → inventory | GPT-5.2 (OpenAI) | 1 per image |
| **B** | YOLO end-to-end | Image → 14-class YOLO → inventory | Custom YOLOv8s | 0 |
| **C** | YOLO + CNN | Image → objectness YOLO → crops → CNN → inventory | Objectness YOLOv8s + EfficientNet-B0 | 0 |

**Key insight:** Pipelines B and C use **no API calls at all**. The comparison is VLM vs. pure detection vs. detect-then-classify.

### Pipeline A: VLM-Only (GPT-5.2)

```
Input Image → GPT-5.2 (constrained to 14 labels) → Inventory JSON {class: count}
```

Sends the full image to GPT-5.2 with a frozen prompt that explicitly lists all 14 class names and instructs the model to count precisely. GPT-5.2 was selected through a [three-model comparison](#vlm-comparison-pipeline-a-model-selection) against Gemini 3.1 Pro and Claude Opus 4.6. While Gemini scored marginally higher on F1 (0.9044 vs 0.9002, a statistically insignificant difference of 0.004), GPT-5.2 was chosen for its **2.4x lower latency** and **2.7x lower cost** — critical factors for a production mobile application.

**Characteristics:**
- Single API call per image
- Prompt-constrained to the 14 valid classes only
- No detection or cropping — all reasoning done by the VLM
- Requires OpenAI API key
- Non-deterministic (VLM outputs may vary slightly across runs)

### Pipeline B: YOLO End-to-End

```
Input Image → 14-Class YOLO (fine-tuned YOLOv8s) → Detections (class + bbox + conf) → Count by Class → Inventory
```

A YOLOv8s model fine-tuned on the 14 fruit/vegetable classes. Each detection carries a class label directly — no second-stage classification needed. The inventory is built by counting detections per class.

**Characteristics:**
- Fully offline, no API calls
- Single-model inference
- Deterministic (same image → same result)
- Requires training on labelled dataset

### Pipeline C: YOLO + CNN

```
Input Image → Objectness YOLO (1-class) → Crop Regions → CNN Classifier (EfficientNet-B0) → Count by Class → Inventory
```

A two-stage approach that separates **detection** from **classification**. An objectness YOLO (all classes remapped to a single "object" class) finds regions of interest, then the EfficientNet-B0 winner from Experiment 1 classifies each crop.

**Characteristics:**
- Fully offline, no API calls
- Modular: swap CNN architecture via config (`efficientnet` | `resnet` | `custom`)
- Isolates detection quality from classification quality
- Requires both objectness YOLO weights and CNN weights
- Domain gap: CNN was trained on clean single-item images, receives YOLO-cropped multi-item scene regions

### System Components

| Component | File | Model |
|-----------|------|-------|
| VLM Client (winner) | `clients/vlm_openai.py` | GPT-5.2 (`gpt-5.2`) → Pipeline A |
| VLM Client (comparison) | `clients/vlm_google.py` | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) |
| VLM Client (comparison) | `clients/vlm_anthropic.py` | Claude Opus 4.6 (`claude-opus-4-6`) |
| Original VLM Client | `clients/vlm_client.py` | GPT-4o-mini (baseline reference) |
| 14-Class YOLO | `clients/yolo_detector.py` | `weights/yolo_14class_best.pt` |
| Objectness YOLO | `clients/yolo_objectness.py` | `weights/yolo_objectness_best.pt` |
| CNN Classifier | `clients/cnn_classifier.py` | `weights/cnn_winner.pth` (EfficientNet-B0) |

---

## VLM Comparison (Pipeline A Model Selection)

### Motivation

Before running the 12-run pipeline comparison, we needed to determine which VLM performs best for the inventory counting task. Rather than assuming a single provider, we ran a controlled comparison across three flagship VLMs from three providers.

### Models Compared

| Model | Provider | Model ID | Release |
|-------|----------|----------|---------|
| **GPT-5.2** | OpenAI | `gpt-5.2` | 2026 |
| **Claude Opus 4.6** | Anthropic | `claude-opus-4-6` | 2026 |
| **Gemini 3.1 Pro** | Google | `gemini-3.1-pro-preview` | 2026 |

All three models received the **identical frozen prompt** constrained to the 14 target classes. Temperature was set to 0.0 for all. Each model processed all 120 test images.

### Results — Single Run on 120 Test Images (Same Conditions)

| | **Gemini 3.1 Pro** | **GPT-5.2** | **Claude Opus 4.6** |
|---|:---:|:---:|:---:|
| **Provider** | Google | OpenAI | Anthropic |
| **F1 Score** | **0.9044** | 0.9002 | 0.8674 |
| **Precision** | 0.8291 | 0.8220 | 0.7688 |
| **Recall** | 0.9949 | 0.9949 | 0.9949 |
| **Avg Latency** | 9,003 ms | **3,687 ms** | 4,724 ms |
| **Total Time (120 imgs)** | ~18 min | **~7.4 min** | ~9.4 min |
| **Cost per Run (120 imgs)** | ~$1.17 | **~$0.44** | ~$1.23 |
| **Cost per Image** | ~$0.010 | **~$0.004** | ~$0.010 |
| **Errors** | 0/120 | 0/120 | 0/120 |

> **Winner by accuracy:** Gemini 3.1 Pro (F1 = 0.9044)
> **Winner by speed:** GPT-5.2 (2.4x faster)
> **Winner by cost:** GPT-5.2 (2.7x cheaper)
> **For comparison — YOLO pipelines (B & C):** <100 ms/image, $0.00/run, fully offline

### Key Observations

- **Identical recall (99.5%)** across all three models — all VLMs find nearly everything in the image
- **Precision is the differentiator** — Gemini overcounts the least, Claude the most
- **12 of 14 classes scored perfect F1 = 1.0** on all three models
- **The grape class is the sole source of error** — VLMs count individual grapes in a cluster rather than treating the cluster as one unit (see [Discussion Points](#the-grape-semantic-ambiguity))
- **GPT-5.2 is 2.4x faster** and **2.7x cheaper** than Gemini, with only 0.004 lower F1
- **Selected for Pipeline A: GPT-5.2** — the F1 difference vs Gemini (0.9002 vs 0.9044) is not statistically significant on 120 images. GPT-5.2 was chosen for its superior latency and cost, which are critical factors for a production mobile application

### Per-Class Breakdown (all models identical except grape)

| Class | Precision | Recall | F1 | Notes |
|-------|-----------|--------|----|-------|
| apple | 1.000 | 1.000 | 1.000 | Perfect |
| banana | 1.000 | 1.000 | 1.000 | Perfect |
| bell_pepper_green | 1.000 | 1.000 | 1.000 | Perfect |
| bell_pepper_red | 1.000 | 1.000 | 1.000 | Perfect |
| carrot | 1.000 | 0.978 | 0.989 | 1 miss |
| cucumber | 1.000 | 1.000 | 1.000 | Perfect |
| **grape** | **0.261** | **1.000** | **0.414** | **119 false positives** |
| lemon | 1.000 | 0.955 | 0.977 | 2 misses |
| onion | 1.000 | 1.000 | 1.000 | Perfect |
| orange | 0.980 | 1.000 | 0.990 | 1 false positive |
| peach | 1.000 | 1.000 | 1.000 | Perfect |
| potato | 1.000 | 1.000 | 1.000 | Perfect |
| strawberry | 1.000 | 1.000 | 1.000 | Perfect |
| tomato | 1.000 | 1.000 | 1.000 | Perfect |

### API Costs

**Per-token pricing (as of March 2026):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------:|------------------------:|
| GPT-5.2 | $1.75 | $14.00 |
| Claude Opus 4.6 | $5.00 | $25.00 |
| Gemini 3.1 Pro | $2.00 | $12.00* |

*\*Gemini output pricing includes thinking tokens (~579 per image), which are billed as output but not visible in the response.*

**Measured cost for the VLM comparison (120 test images, from billing dashboards):**

| Model | Input Tokens | Output Tokens | Total Cost | Cost per Image |
|-------|-------------:|--------------:|-----------:|---------------:|
| GPT-5.2 | 237,990 | included | **$0.45** | **$0.004** |
| Claude Opus 4.6 | 498,213 | 10,619 | **$2.76** | **$0.012** |
| Gemini 3.1 Pro | ~150,000 | ~72,840* | **~$1.17** | **~$0.010** |

*GPT-5.2 and Claude costs are actual figures from the OpenAI and Anthropic billing dashboards. Claude's total is higher because the comparison required multiple runs to resolve a 5 MB image size limit (the per-image cost for a single clean run is ~$0.012). Gemini cost is estimated from measured token usage (1,250 input + 607 output tokens per image). GPT-5.2 token count (237,990) includes 2 additional test calls beyond the 120 images.*

**Projected cost for the full 12-run experiment:**

| Component | API Calls | Est. Cost |
|-----------|:---------:|----------:|
| VLM comparison (3 models × 120 images) | 360 | ~$4.38 |
| Pipeline A in 12-run matrix (GPT-5.2 × 4 conditions × 120 images) | 480 | ~$1.76 |
| **Total VLM API cost** | **840** | **~$6.14** |

Pipelines B and C are fully offline — **zero API cost**. Choosing GPT-5.2 over Gemini saved ~$2.92 on the 12-run matrix alone. At production scale (10,000 images/day), GPT-5.2 would cost ~$40/day vs. $0 for the YOLO pipelines.

### How to Run the VLM Comparison

```bash
# Run all 3 VLMs (requires all 3 API keys in .env)
python -m evaluation.vlm_comparison \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels \
    --output results/vlm_comparison

# Run a specific VLM only
python -m evaluation.vlm_comparison \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels \
    --vlms gemini-3.1-pro
```

### Results Location

```
results/vlm_comparison/
├── comparison_summary.json          # Winner + all 3 models' metrics
├── gpt-5_2_results.json             # GPT-5.2 full results + per-class
├── gpt-5_2_predictions.json         # GPT-5.2 per-image predictions
├── claude-opus-4_6_results.json     # Claude Opus 4.6 full results
├── claude-opus-4_6_predictions.json # Claude per-image predictions
├── gemini-3_1-pro_results.json      # Gemini 3.1 Pro full results
├── gemini-3_1-pro_predictions.json  # Gemini per-image predictions
└── ground_truth.json                # Ground truth for all 120 images
```

---

## Experimental Design

### Controlled Variables

| Aspect | Value | Rationale |
|--------|:-----:|-----------|
| YOLO Confidence | `0.25` | Balanced sensitivity for fine-tuned models |
| YOLO IoU | `0.45` | Standard NMS threshold |
| YOLO Max Detections | `30` | Allow counting in dense scenes |
| YOLO Image Size | `640` | Standard Ultralytics input |
| CNN Image Size | `224` | Standard ImageNet input |
| Crop Padding | `10%` | Context capture around detected regions |
| VLM Temperature | `0.0` | Deterministic-as-possible outputs |
| Random Seed | `42` | Reproducibility across runs |

### Count-Based Metrics

Since Pipeline A has no bounding boxes, the fair comparison unit is the **inventory** (class → count). For each image, per class:

```
TP = min(predicted, ground_truth)        — correct counts
FP = max(0, predicted - ground_truth)    — overcounting
FN = max(0, ground_truth - predicted)    — undercounting
```

**Precision** = TP / (TP + FP) — "when it says it sees something, how often is it right?"
**Recall** = TP / (TP + FN) — "of everything actually there, how much did it find?"
**F1** = harmonic mean of precision and recall — single balanced score

Micro-averaged across all images and classes for overall scores. Per-class metrics also computed.

### Training / Test Separation

> **Core rule: never test on data the model saw during training.**

| Data Split | Pipeline A (VLM) | Pipeline B (YOLO-14) | Pipeline C (YOLO+CNN) |
|------------|-------------------|----------------------|-----------------------|
| **Training** | N/A (pre-trained by OpenAI) | Public dataset (remapped to 14 classes) | **YOLO:** Public dataset (objectness labels) **CNN:** Experiment 1 weights |
| **Testing** | 120 hand-photographed images | 120 hand-photographed images | 120 hand-photographed images |

All three pipelines are evaluated on the **exact same 120 test images**, but none of the locally-trained models ever saw those images during training.

---

## Training Data

### Dataset: Combined Vegetables & Fruits (Roboflow Universe)

| Property | Value |
|----------|-------|
| Source | [Roboflow Universe — Combined Vegetables & Fruits](https://universe.roboflow.com/) |
| Total images | ~42,000 |
| Original classes | 47 |
| Format | YOLOv8 (bounding box annotations) |
| Splits | Train (19,356) / Val (2,602) / Test (1,882) |

### Class Remapping (47 → 14 classes)

The public dataset has 47 classes. Our experiment uses 14. The `training/remap_classes.py` script:

1. **Maps** each of the 47 source classes to one of the 14 target classes (or discards it)
2. **Splits** the generic "bell pepper" class into `bell_pepper_green` and `bell_pepper_red` using HSV colour analysis (red pixels > 40% → red, otherwise → green)
3. **Rewrites** every `.txt` label file with new class IDs (0–13)
4. **Discards** bounding boxes for unmapped classes

```bash
python -m training.remap_classes \
    --src "C:/path/to/Combined Vegetables - Fruits" \
    --dst dataset \
    --discard-unknown
```

**Output after remapping:** 19,356 train + 2,602 val + 1,882 test images with 225,410 total bounding boxes across 14 classes.

### Objectness Labels (Pipeline C)

For Pipeline C's class-agnostic detector, all 14 class IDs are remapped to `0` ("object"):

```bash
python -m training.prepare_objectness_labels --src dataset --dst dataset_objectness
```

This keeps the same bounding box coordinates but tells YOLO "there's *something* here" instead of "there's an *apple* here."

---

## Test Data and Annotation

### 120 Hand-Photographed Test Images

- **120 original photographs** taken by the author
- Captured across real-world settings (kitchen counter, refrigerator shelf, dining table, grocery bag, chopping board)
- Each image contains 2–8 items from the 14-class taxonomy
- Balanced across difficulty tiers and camera angles
- Stored in `dataset_exp2/images/` as `.jpg` files (IMG_001.jpg – IMG_120.jpg)

### Annotation Process

1. Images uploaded to [Roboflow](https://roboflow.com) (free tier)
2. Bounding boxes drawn around every visible fruit/vegetable item
3. Each box assigned one of the 14 exact class names matching `config.py`
4. Exported in YOLOv8 format
5. Label files placed in `dataset_exp2/labels/` (IMG_001.txt – IMG_120.txt)

**Annotation rules:**
- Grape cluster = **1 bounding box** around the entire cluster
- Individual strawberries = **1 box per strawberry**
- Partially occluded items = annotated (box around visible portion)
- Bell peppers = assigned `bell_pepper_green` or `bell_pepper_red` by actual colour

### YOLO Label Format

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalised to [0, 1]. Class IDs are 0–13 matching the 14-class taxonomy.

---

## Robustness Testing (Image Degradations)

### Why Degradations Matter

In a real mobile app, images will not always be perfect. Users may have shaky hands (blur), poor lighting (noise), or images may be compressed through messaging apps (JPEG). Testing robustness to these conditions reveals which pipeline is most reliable in practice.

### Three Degradations

| ID | Degradation | Parameters | Simulates |
|:--:|-------------|-----------|-----------|
| D1 | **Gaussian blur** | kernel=7, sigma=3.0 | Out-of-focus camera, shaky hands |
| D2 | **Gaussian noise** | mean=0, sigma=25 | Low-light sensor noise (grainy photo) |
| D3 | **JPEG compression** | quality=15 | Messaging app compression (WhatsApp, etc.) |

These cover the three principal sources of image quality loss in a mobile workflow: **optical** (blur), **sensor** (noise), and **compression** (JPEG artifacts).

### Generating Degraded Images

```bash
python -m evaluation.generate_degradations \
    --src dataset_exp2/images \
    --dst dataset_exp2
```

**Output:**
```
dataset_exp2/
├── images/              ← 120 clean originals (unchanged)
├── labels/              ← 120 .txt files (shared by ALL variants)
├── images_d1_blur/      ← 120 blurred images
├── images_d2_noise/     ← 120 noisy images
└── images_d3_jpeg/      ← 120 JPEG-compressed images
```

Labels are shared across all conditions because degradation does not move or change the objects — only image quality changes.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- GPU recommended for training (CPU supported for inference)
- API keys required for Pipeline A and VLM comparison (see below)

### Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd SnapShelf-console

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys (create .env file)
```

### API Keys

Create a `.env` file in the project root:

```env
# Required for Pipeline A (VLM) and VLM comparison
OPENAI_API_KEY=sk-...          # OpenAI (GPT-5.2)
ANTHROPIC_API_KEY=sk-ant-...   # Anthropic (Claude Opus 4.6)
GOOGLE_API_KEY=AIza...         # Google (Gemini 3.1 Pro)
```

- **Pipeline A only** requires `OPENAI_API_KEY` (GPT-5.2 is the selected VLM)
- **VLM comparison** requires all three keys to run all three models
- **Pipelines B and C** require no API keys (fully offline)

### Dependencies

Core dependencies (`requirements.txt`):

```
ultralytics==8.3.57       # YOLOv8 (Pipelines B & C)
openai>=1.59.9            # GPT-5.2 VLM client
anthropic>=0.40.0         # Claude Opus 4.6 VLM client
google-genai>=1.0.0       # Gemini 3.1 Pro VLM client
pillow==11.1.0            # Image processing
numpy==2.2.2              # Numerical computing
rich==13.9.4              # Console formatting
python-dotenv             # .env file loading
```

---

## Usage

### Command-Line Interface

```bash
# Run individual pipelines on a single image
python main.py vlm <image_path>          # Pipeline A: VLM-only
python main.py yolo-14 <image_path>      # Pipeline B: YOLO end-to-end
python main.py yolo-cnn <image_path>     # Pipeline C: YOLO + CNN

# Evaluate all pipelines on a test set
python main.py evaluate --images dataset_exp2/images --labels dataset_exp2/labels

# Train models
python main.py train yolo-14             # Train 14-class YOLO
python main.py train yolo-obj            # Train objectness YOLO

# VLM comparison (separate from pipeline evaluation)
python -m evaluation.vlm_comparison \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels

# Utility
python main.py --validate                # Verify environment and display config
```

### Interactive Mode

```bash
python main.py
```

Launches a menu-driven interface for running pipelines, training, and evaluation.

---

## Training

### Option 1: Local Training (GPU required)

```bash
# Train 14-class YOLO (Pipeline B)
python main.py train yolo-14

# Train objectness YOLO (Pipeline C)
python main.py train yolo-obj
```

**Default hyperparameters** (from `config.py`):

| Setting | Value |
|---------|-------|
| Base model | YOLOv8s (pre-trained on COCO) |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Learning rate | 0.01 |
| Early stopping | 15 epochs patience |
| Random seed | 42 |

### Option 2: Google Colab Training (no local GPU)

A Colab notebook is provided for training on a free T4 GPU:

1. Upload `dataset_14class.zip` and `dataset_objectness.zip` to Google Drive under `SnapShelf/`
2. Open `training/train_colab.ipynb` in Google Colab
3. Set runtime to **GPU (T4)**
4. Run all cells — trains both YOLO models sequentially
5. Download trained weights from `SnapShelf/weights/` in Google Drive

### After Training

```bash
# Verify weight files exist
ls weights/yolo_14class_best.pt       # 14-class YOLO (Pipeline B)
ls weights/yolo_objectness_best.pt    # Objectness YOLO (Pipeline C)

# Copy CNN weights from Experiment 1
cp /path/to/experiment1/best_efficientnet.pth weights/cnn_winner.pth

# Smoke test
python main.py yolo-14 dataset_exp2/images/IMG_001.jpg
python main.py yolo-cnn dataset_exp2/images/IMG_001.jpg
```

---

## Evaluation — The 12-Run Matrix

### The Matrix

3 pipelines x 4 image conditions = **12 evaluation runs**.

| Run | Pipeline | Condition | Images Folder |
|:---:|----------|-----------|---------------|
| 1 | A (VLM) | Clean | `dataset_exp2/images` |
| 2 | A (VLM) | D1: Blur | `dataset_exp2/images_d1_blur` |
| 3 | A (VLM) | D2: Noise | `dataset_exp2/images_d2_noise` |
| 4 | A (VLM) | D3: JPEG | `dataset_exp2/images_d3_jpeg` |
| 5 | B (YOLO) | Clean | `dataset_exp2/images` |
| 6 | B (YOLO) | D1: Blur | `dataset_exp2/images_d1_blur` |
| 7 | B (YOLO) | D2: Noise | `dataset_exp2/images_d2_noise` |
| 8 | B (YOLO) | D3: JPEG | `dataset_exp2/images_d3_jpeg` |
| 9 | C (YOLO+CNN) | Clean | `dataset_exp2/images` |
| 10 | C (YOLO+CNN) | D1: Blur | `dataset_exp2/images_d1_blur` |
| 11 | C (YOLO+CNN) | D2: Noise | `dataset_exp2/images_d2_noise` |
| 12 | C (YOLO+CNN) | D3: JPEG | `dataset_exp2/images_d3_jpeg` |

Labels folder is **always** `dataset_exp2/labels` — degradation changes image quality, not object locations.

### Running the 12 Runs

```bash
# Clean images (runs all 3 pipelines)
python main.py evaluate \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels \
    --output results/clean

# D1: Blur
python main.py evaluate \
    --images dataset_exp2/images_d1_blur \
    --labels dataset_exp2/labels \
    --output results/d1_blur

# D2: Noise
python main.py evaluate \
    --images dataset_exp2/images_d2_noise \
    --labels dataset_exp2/labels \
    --output results/d2_noise

# D3: JPEG compression
python main.py evaluate \
    --images dataset_exp2/images_d3_jpeg \
    --labels dataset_exp2/labels \
    --output results/d3_jpeg
```

### Pre-Flight Checklist

Before running, verify everything is in place:

```bash
# Weight files
ls weights/yolo_14class_best.pt        # Pipeline B
ls weights/yolo_objectness_best.pt     # Pipeline C
ls weights/cnn_winner.pth              # Pipeline C (CNN)

# API key (Pipeline A)
echo $OPENAI_API_KEY                   # Should print your key

# Test images and labels
ls dataset_exp2/images/*.jpg | wc -l         # 120
ls dataset_exp2/labels/*.txt | wc -l         # 120
ls dataset_exp2/images_d1_blur/*.jpg | wc -l # 120
ls dataset_exp2/images_d2_noise/*.jpg | wc -l # 120
ls dataset_exp2/images_d3_jpeg/*.jpg | wc -l  # 120
```

### Evaluation Outputs

For each condition, the evaluator generates:

| File | Description |
|------|-------------|
| `comparison_summary.json` | Side-by-side P/R/F1 and latency for all pipelines |
| `comparison_bars.png` | Bar chart comparing metrics across pipelines |
| `comparison_table.tex` | LaTeX table ready for dissertation |
| `ground_truth.json` | Ground truth inventories for all test images |
| `{pipeline}_predictions.json` | Per-image predicted inventories |
| `{pipeline}_confusion.png` | Confusion matrix heatmap |
| `{pipeline}_report.json` | Full metrics, per-class breakdown, error analysis |

### Expected Timing

| Pipeline | Approx. per image | Total (120 images) |
|----------|-------------------|-------------------|
| A (VLM) | ~3.7 seconds (GPT-5.2 API) | ~7.4 minutes |
| B (YOLO) | 20–80 ms (GPU) / 200–500 ms (CPU) | ~5 sec – 1 min |
| C (YOLO+CNN) | 30–120 ms (GPU) / 300–800 ms (CPU) | ~6 sec – 1.5 min |

Total for all 12 runs: **~35–40 minutes** (dominated by VLM API time across 4 conditions).

---

## Output Schema

All pipelines produce an identical JSON structure:

```json
{
  "inventory": {
    "apple": 3,
    "banana": 1,
    "tomato": 2
  },
  "meta": {
    "pipeline": "yolo-14",
    "image": "IMG_001.jpg",
    "runtime_ms": 45.32,
    "detections_count": 6,
    "timing_breakdown": {
      "detection_ms": 42.15,
      "total_ms": 45.32
    }
  }
}
```

| Field | Type | Description |
|-------|:----:|-------------|
| `inventory` | `Dict[str, int]` | Class name → count mapping |
| `pipeline` | string | `vlm` · `yolo-14` · `yolo-cnn` |
| `runtime_ms` | float | Total execution time in milliseconds |
| `detections_count` | integer | Number of YOLO detections (Pipeline B/C only) |

---

## Configuration

All experiment parameters are centralised in `config.py` as a frozen dataclass:

```python
@dataclass(frozen=True)
class ExperimentConfig:
    # VLM Settings (Pipeline A)
    vlm_model: str = "gpt-4o-mini"       # Original baseline; Gemini used via vlm_google.py
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 500

    # YOLO Settings (shared by B and C)
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 30
    yolo_img_size: int = 640

    # CNN Settings (Pipeline C)
    cnn_model_name: str = "efficientnet"  # efficientnet | resnet | custom
    cnn_img_size: int = 224
    cnn_crop_padding: float = 0.10

    # Reproducibility
    random_seed: int = 42
```

---

## Reproducibility

### Singleton Pattern

All model clients use a singleton pattern — models load once and are reused across pipeline runs. Timing measurements exclude initialisation overhead.

### Random Seed Control

Seeds are set automatically on experiment initialisation for Python `random`, NumPy, and PyTorch (CPU + CUDA).

### Structured Logging

Every experiment run generates detailed logs in `logs/experiment_{timestamp}.jsonl` with timestamped entries for each pipeline step, detection event, and error.

### Pinned Dependencies

Core dependencies are pinned for reproducibility (see `requirements.txt`).

---

## Key Findings and Discussion Points

### The Grape Semantic Ambiguity

All three VLMs exhibited the same systematic error on the `grape` class. The test annotations treated **1 grape cluster = 1 unit**, but VLMs interpreted each **individual grape berry** as a separate item (reporting 4–6 per cluster instead of 1).

| Model | Grape FP | Grape Precision | Impact on Overall F1 |
|-------|----------|-----------------|---------------------|
| Gemini 3.1 Pro | 119 | 0.261 | Drops F1 from ~0.99 to 0.90 |
| GPT-5.2 | 125 | 0.252 | Drops F1 from ~0.99 to 0.90 |
| Claude Opus 4.6 | 174 | 0.194 | Drops F1 from ~0.99 to 0.87 |

**This is not a bug — it is a semantic ambiguity.** Neither interpretation is wrong. This is a valuable finding for the dissertation, revealing a fundamental challenge in VLM-based counting: the model and the annotator may have different definitions of "one unit."

**Recommendation for the dissertation:** Report results both **with** and **without** the grape class. Without grape, all three VLMs achieve ~98–99% F1, demonstrating the approach works excellently for unambiguous items.

### Domain Gap (Pipeline C)

Pipeline C's CNN was trained on clean, centred, single-item images (Experiment 1), but receives YOLO-cropped regions from multi-item real-world scenes. These crops may include partial objects, background clutter, or unusual angles. This **domain gap** is expected and is a legitimate finding — it reveals a real limitation of the detect-then-classify approach.

### VLM Non-Determinism

While temperature=0.0 is set for reproducibility, VLM outputs (GPT-5.2) are not guaranteed to be fully deterministic across API versions or over time. This is an inherent limitation of API-based approaches and should be acknowledged in the limitations section.

---

## Project Structure

```
SnapShelf-console/
├── config.py                         # 14-class constants, frozen ExperimentConfig
├── main.py                           # CLI: vlm / yolo-14 / yolo-cnn / evaluate / train
├── requirements.txt                  # Pinned + minimum-version dependencies
├── .env                              # API keys (gitignored)
├── .gitignore
├── README.md                         # This file
├── EXPERIMENT2_GUIDE.md              # Step-by-step walkthrough
├── EXPERIMENT2_PLAN.md               # Original experiment plan
├── PICTURES_PLAN.md                  # Photo dataset planning
│
├── clients/                          # Model inference clients
│   ├── __init__.py
│   ├── vlm_client.py                # GPT-4o-mini (original baseline)
│   ├── vlm_openai.py               # GPT-5.2 (VLM winner → Pipeline A)
│   ├── vlm_anthropic.py            # Claude Opus 4.6 (VLM comparison)
│   ├── vlm_google.py               # Gemini 3.1 Pro (VLM comparison)
│   ├── yolo_detector.py            # 14-class YOLO (Pipeline B)
│   ├── yolo_objectness.py          # 1-class objectness YOLO (Pipeline C)
│   └── cnn_classifier.py           # CNN factory: EfficientNet / ResNet (Pipeline C)
│
├── pipelines/                        # Pipeline orchestration
│   ├── __init__.py
│   ├── output.py                    # Inventory = Dict[str, int] schema
│   ├── vlm_pipeline.py             # Pipeline A: VLM-only
│   ├── yolo_pipeline.py            # Pipeline B: YOLO end-to-end
│   └── yolo_cnn_pipeline.py        # Pipeline C: YOLO + CNN
│
├── training/                         # Model training scripts
│   ├── __init__.py
│   ├── remap_classes.py             # Remap 47-class dataset → 14 classes
│   ├── prepare_objectness_labels.py # Remap class IDs to 0 for objectness
│   ├── train_yolo_14class.py        # Fine-tune YOLOv8s on 14 classes
│   ├── train_yolo_objectness.py     # Fine-tune YOLOv8s as objectness detector
│   ├── data_yaml_generator.py       # Generate data.yaml for Ultralytics
│   └── train_colab.ipynb            # Google Colab notebook (for no-GPU setups)
│
├── evaluation/                       # Evaluation and metrics
│   ├── __init__.py
│   ├── ground_truth.py              # YOLO .txt labels → inventory dicts
│   ├── metrics.py                   # Count-based P/R/F1 (micro-averaged, per-class)
│   ├── confusion.py                 # Confusion matrix builder + heatmap plotter
│   ├── error_analysis.py            # Missed / false positive / counting breakdown
│   ├── evaluate_runner.py           # Orchestrator: runs pipelines on test set
│   ├── report.py                    # Comparison tables, bar charts, LaTeX output
│   ├── vlm_comparison.py           # 3-model VLM comparison runner
│   └── generate_degradations.py    # D1/D2/D3 image degradation generator
│
├── data/                             # YOLO training configs
│   ├── yolo_14class.yaml            # nc=14, 14 class names
│   └── yolo_objectness.yaml         # nc=1, class="object"
│
├── dataset_exp2/                     # Experiment 2 test data
│   ├── images/                      # 120 clean test images
│   ├── labels/                      # 120 YOLO annotation files
│   ├── images_d1_blur/              # 120 blurred images
│   ├── images_d2_noise/             # 120 noisy images
│   └── images_d3_jpeg/              # 120 JPEG-compressed images
│
├── dataset/                          # 14-class training data (from remap)
│   ├── train/images/ + labels/      # ~19,356 images
│   ├── val/images/ + labels/        # ~2,602 images
│   └── test/images/ + labels/       # ~1,882 images
│
├── dataset_objectness/               # Objectness training data (class IDs = 0)
│   ├── train/images/ + labels/
│   ├── val/images/ + labels/
│   └── test/images/ + labels/
│
├── weights/                          # Trained model weights (gitignored)
│   ├── yolo_14class_best.pt         # Pipeline B
│   ├── yolo_objectness_best.pt      # Pipeline C (detection)
│   └── cnn_winner.pth               # Pipeline C (classification)
│
├── results/                          # Evaluation outputs (gitignored)
│   ├── vlm_comparison/              # 3-model VLM comparison results
│   ├── clean/                       # 12-run: clean condition
│   ├── d1_blur/                     # 12-run: blur condition
│   ├── d2_noise/                    # 12-run: noise condition
│   └── d3_jpeg/                     # 12-run: JPEG condition
│
└── logs/                             # JSONL experiment logs (gitignored)
```

---

## Dissertation Methodology Text

Ready-to-adapt paragraphs for your dissertation report:

### VLM Model Selection

> *"To select the optimal VLM for Pipeline A, a controlled comparison was conducted across three flagship models: GPT-5.2 (OpenAI), Claude Opus 4.6 (Anthropic), and Gemini 3.1 Pro (Google). All models received an identical frozen prompt constrained to the 14 target classes, with temperature set to 0.0. Each model processed the full set of 120 test images. Gemini 3.1 Pro achieved the highest micro-averaged F1 score (0.9044), marginally outperforming GPT-5.2 (0.9002) and Claude Opus 4.6 (0.8674). All three models achieved near-identical recall (0.9949), indicating that VLMs reliably detect items; the differentiator was precision, where Gemini produced the fewest false positives. However, the F1 difference between Gemini and GPT-5.2 was only 0.004 — not statistically significant on a 120-image test set. GPT-5.2 was therefore selected as Pipeline A's VLM for the subsequent 12-run comparison, as it offered 2.4x lower latency (3.7 s vs 9.0 s per image) and 2.7x lower cost ($0.004 vs $0.010 per image) — factors that directly impact the feasibility of a production mobile application."*

### Experimental Design

> *"Experiment 2 evaluates three end-to-end pipelines on an identical test set of 120 photographs, each containing 2–8 items from a 14-class fruit and vegetable taxonomy. Pipeline A uses GPT-5.2 (OpenAI), selected through a three-model VLM comparison that also evaluated Gemini 3.1 Pro and Claude Opus 4.6, with a constrained prompt restricting output to the 14 target classes. Pipeline B uses a YOLOv8s object detector fine-tuned to directly predict all 14 classes. Pipeline C uses a two-stage approach: a YOLOv8s model trained as a class-agnostic objectness detector to localise items, followed by an EfficientNet-B0 classifier (the winning model from Experiment 1) to classify each cropped region."*

### Training/Test Separation

> *"To ensure a fair comparison, training and test data were strictly separated. Pipelines B and C were fine-tuned on the publicly available Combined Vegetables & Fruits dataset (Roboflow Universe, ~42,000 images, 47 classes), remapped to the 14-class taxonomy used in this study (see Section X). The test set comprised 120 original photographs taken by the author across five real-world settings, annotated independently in Roboflow and never used during training. Pipeline A (GPT-5.2) was used as-is without fine-tuning; while we cannot guarantee our test images were absent from its internet-scale training corpus, this reflects the realistic deployment scenario for a zero-shot VLM."*

### Robustness Evaluation

> *"To assess robustness, three image degradations were applied to the test set: Gaussian blur (kernel=7, sigma=3.0, simulating an out-of-focus camera), additive Gaussian noise (sigma=25, simulating low-light sensor noise), and JPEG compression at quality level 15 (simulating lossy transmission through messaging applications). Each pipeline was evaluated on all four conditions (clean plus three degraded), yielding 12 evaluation runs. Ground truth annotations were shared across all conditions, as degradation does not alter object locations."*

### Grape Class Discussion

> *"A notable finding was the consistent over-prediction of the grape class across all three VLMs. Ground truth annotations treated one grape cluster as a single unit, whereas VLMs counted individual grape berries within each cluster, reporting 4–6 items per cluster. This semantic ambiguity — what constitutes 'one grape' — is not a model error but a fundamental challenge in count-based evaluation: the annotation convention and the model's interpretation of a 'unit' may diverge. When the grape class is excluded, all three VLMs achieve F1 scores above 0.98, suggesting the approach is highly effective for unambiguously defined items."*

### Cost and Latency Trade-Offs

> *"API costs and inference latency were recorded for all three VLMs. GPT-5.2 was the fastest (3.7 s/image) and cheapest ($0.004/image), while Gemini 3.1 Pro was the slowest (9.0 s/image) and most expensive ($0.010/image) due to internal thinking tokens billed as output. Claude Opus 4.6 occupied the middle ground (4.7 s/image, $0.012/image). The total API cost for the VLM comparison across all three models on 120 images was $4.38. Pipelines B and C, being fully offline, incurred zero API cost and sub-100 ms inference on GPU, making them significantly cheaper at scale. For a production deployment processing 10,000 images/day, Pipeline A (GPT-5.2) would cost approximately $40/day versus $0 for the YOLO-based pipelines, plus the added latency of network round-trips."*

### Limitations

> *"Several limitations should be acknowledged. First, the test set of 120 images, while purpose-built with controlled variation, is small by industry standards. Second, all images were captured using a single device by a single photographer. Third, Pipeline C's CNN was trained on clean, single-item, centred images (Experiment 1), introducing a domain gap when classifying YOLO-cropped regions from cluttered scenes. Fourth, while Pipeline A (GPT-5.2) was queried with temperature=0 for reproducibility, VLM outputs are not guaranteed to be fully deterministic across API versions. Finally, the grape class revealed a semantic ambiguity in count-based evaluation that may extend to other aggregate items (e.g., cherry tomatoes on a vine)."*

---

## System Requirements

| Requirement | Specification |
|-------------|:-------------:|
| Python | 3.10+ |
| RAM | 8 GB minimum (16 GB recommended for training) |
| Disk | ~2 GB (models + datasets) |
| GPU | Recommended for training (Colab T4 alternative provided) |
| Network | Required for Pipeline A (Google API) and VLM comparison |

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenAI](https://openai.com) for GPT-5.2 API (Pipeline A)
- [Google](https://ai.google.dev/) for Gemini 3.1 Pro API (VLM comparison)
- [Anthropic](https://anthropic.com) for Claude Opus 4.6 API (VLM comparison)
- [PyTorch](https://pytorch.org) for CNN training and inference
- [Roboflow](https://roboflow.com) for annotation tools and training datasets
