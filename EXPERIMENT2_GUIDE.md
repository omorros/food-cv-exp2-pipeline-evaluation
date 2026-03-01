# Experiment 2 — Step-by-Step Guide

> **What this document is:** A beginner-friendly, end-to-end walkthrough for
> running the full 12-run pipeline comparison. Follow each section in order.
> Every command is copy-pasteable.

---

## Table of Contents

1. [Why You Need Separate Data (The Fairness Problem)](#1-why-you-need-separate-data-the-fairness-problem)
2. [Download the Public Training Dataset](#2-download-the-public-training-dataset)
3. [Remap Classes to Your 14-Class System](#3-remap-classes-to-your-14-class-system)
4. [Generate Objectness Labels for Pipeline C](#4-generate-objectness-labels-for-pipeline-c)
5. [Annotate Your 120 Test Images in Roboflow](#5-annotate-your-120-test-images-in-roboflow)
6. [Train the YOLO Models](#6-train-the-yolo-models)
7. [Set Up Pipeline C's CNN (EfficientNet from Experiment 1)](#7-set-up-pipeline-cs-cnn-efficientnet-from-experiment-1)
8. [Generate Degraded Image Variants](#8-generate-degraded-image-variants)
9. [Run All 12 Evaluation Runs](#9-run-all-12-evaluation-runs)
10. [Dissertation Methodology Text](#10-dissertation-methodology-text)

---

## 1. Why You Need Separate Data (The Fairness Problem)

### How YOLO actually learns (the big picture)

YOLO learning happens in two completely separate phases:

**Phase 1 — Training (the textbook).** You show YOLO thousands of images with
bounding boxes already drawn and say "this is an apple, this is a banana." YOLO
adjusts its internal weights until it learns visual patterns: "round + red =
apple", "yellow + elongated = banana." After training, YOLO has never seen your
real-world test photos — it only studied the textbook.

**Phase 2 — Testing (the exam).** You show YOLO images it has **never seen
before** and ask "what do you see?" You compare its answers to the correct
answers (your annotations) to measure how good it really is.

This means you need **two different sets of images** that do **two different
jobs:**

| Step | Which images? | Who draws the bounding boxes? | Purpose |
|------|--------------|------------------------------|---------|
| Training | Public Kaggle dataset (~15,000 images) | **Already done for you** — the dataset comes pre-annotated | Teach YOLO what fruits and vegetables look like |
| Testing | Your 120 original photos | **You** annotate them in Roboflow (Section 5) | Measure how well YOLO performs on images it has never seen |

Think of it like a driving test:
- **Training** = the practice routes you drive during lessons (Kaggle dataset)
- **Testing** = the route the examiner picks on test day — one you've never driven before (your 120 photos)
- **Answer key** = the examiner's clipboard with the correct turns (your Roboflow annotations)

If YOLO passes the exam (performs well on your 120 unseen photos), you can
trust it will also handle future unseen photos in the real app. If you let it
cheat by testing on training images, a 99% score tells you nothing.

### The core rule

> **Never test on data the model saw during training.**

If a YOLO model trains on a photo of three apples on a counter, and you then
"test" it on that same photo, it will score near-perfectly — not because it
learned to detect apples, but because it memorised that specific image. The
resulting F1 score is meaningless.

Pipeline A (VLM) uses GPT-5.2 (selected from a
[three-model VLM comparison](#vlm-comparison) for its optimal balance of
accuracy, speed, and cost), which was pre-trained by OpenAI on internet data. You cannot guarantee your images were never in its training
set, but you **can** guarantee that your own YOLO and CNN models never saw the
test images.

### What each pipeline uses

| Data Split | Pipeline A (VLM) | Pipeline B (YOLO-14) | Pipeline C (YOLO+CNN) |
|------------|-------------------|----------------------|-----------------------|
| **Training** | N/A (pre-trained by OpenAI) | Public dataset (remapped to 14 classes) | **YOLO:** Public dataset (objectness labels) **CNN:** Experiment 1 weights (single-item images) |
| **Testing** | Your 120 hand-photographed images | Your 120 hand-photographed images | Your 120 hand-photographed images |

The critical point: **all three pipelines are evaluated on the exact same 120
test images**, but none of the locally-trained models (YOLO-14, objectness YOLO,
CNN) ever saw those images during training.

### Why we use a public dataset for training

You photographed 120 images for testing. You need *different* images for
training YOLO. Rather than photographing hundreds more images yourself, you
download a large public fruit/vegetable dataset that **already has bounding-box
annotations** — no Roboflow work needed for training data. This gives you:

- **Volume:** Thousands of labelled images (vs. your 120)
- **Separation:** Zero overlap with your test set
- **Reproducibility:** Anyone can download the same dataset and replicate your results
- **No extra annotation work:** The bounding boxes are included in the download

### Dissertation justification (adapt for your methodology chapter)

> *"To ensure a fair comparison, training and test data were strictly separated.
> Pipelines B and C were fine-tuned on the publicly available
> Fruits-And-Vegetables-Detection-Dataset (Kaggle), which contains annotated
> bounding boxes across 63 fruit and vegetable classes. These were remapped to
> our 14-class taxonomy (see Section X). The test set comprised 120 original
> photographs taken in five real-world settings, annotated independently and
> never used during training. Pipeline A (GPT-5.2) was used as-is without
> fine-tuning; while we cannot guarantee our test images were absent from its
> internet-scale training corpus, this reflects the realistic deployment scenario
> for a zero-shot VLM."*

---

## 2. Download the Public Training Dataset

### Which dataset

**LVIS Fruits And Vegetables Detection Dataset** — 63 classes, YOLO format.

- GitHub: `https://github.com/henningheyen/Fruits-And-Vegetables-Detection-Dataset`
- Kaggle: `https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset`

This dataset contains **~8,200 images** with YOLO-format bounding-box
annotations across **63 classes** of fruits and vegetables. It includes
train/val splits (6,721 train + 1,500 val).

### How to download

**Option A — Kaggle web UI (easiest)**

1. Go to the URL above
2. Click **Download** (top-right). You will need a free Kaggle account.
3. A `.zip` file will download (~2 GB)
4. Extract it into a folder **outside** your project, e.g.:
   ```
   C:\Users\YourName\datasets\fruits-and-vegetables-detection\
   ```

**Option B — Kaggle CLI**

```bash
# Install the Kaggle CLI (one-time)
pip install kaggle

# Set up your API key:
# 1. Go to https://www.kaggle.com/settings → "Create New Token"
# 2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json

# Download
kaggle datasets download -d ajaygorkar/fruits-and-vegetables-detection-dataset

# Extract
unzip fruits-and-vegetables-detection-dataset.zip -d ../fruits-veg-dataset
```

### Expected folder structure after extraction

```
fruits-and-vegetables-detection/
├── train/
│   ├── images/          ← ~11,000 .jpg files
│   └── labels/          ← ~11,000 .txt files (YOLO format)
├── val/
│   ├── images/          ← ~2,500 .jpg files
│   └── labels/          ← ~2,500 .txt files
└── test/
    ├── images/          ← ~1,500 .jpg files
    └── labels/          ← ~1,500 .txt files
```

### Verify the download

Open a terminal and check:

```bash
# Count images in each split
ls train/images/*.jpg | wc -l
ls val/images/*.jpg | wc -l
ls test/images/*.jpg | wc -l

# Check a sample label file
cat train/labels/$(ls train/labels/ | head -1)
```

Each label line should look like: `<class_id> <x_center> <y_center> <width> <height>`

All coordinates should be between 0 and 1. Class IDs will be 0–62 (63 classes).

---

## 3. Remap Classes to Your 14-Class System

### Why remapping is needed

The public dataset has **63 classes**. Your experiment uses **14 classes**. You
need to:

1. **Map** each of the 63 source classes to one of your 14 target classes (or discard it)
2. **Split** the generic "bell pepper" / "capsicum" class into `bell_pepper_green` and `bell_pepper_red` using colour analysis
3. **Rewrite** every `.txt` label file with the new class IDs

### Your 14 target classes

| ID | Class | ID | Class |
|:--:|-------|:--:|-------|
| 0 | `apple` | 7 | `lemon` |
| 1 | `banana` | 8 | `onion` |
| 2 | `bell_pepper_green` | 9 | `orange` |
| 3 | `bell_pepper_red` | 10 | `peach` |
| 4 | `carrot` | 11 | `potato` |
| 5 | `cucumber` | 12 | `strawberry` |
| 6 | `grape` | 13 | `tomato` |

### The bell pepper green/red colour split

The public dataset typically labels all bell peppers under one class (e.g.
"capsicum" or "bell pepper"). Your taxonomy splits them by colour. The
`remap_classes.py` script handles this by:

1. Cropping the bounding box region from the image
2. Converting the crop to HSV colour space
3. Computing the percentage of red-ish pixels (hue 0–10 or 170–180)
4. If red pixels > 40% → `bell_pepper_red` (class 3), otherwise → `bell_pepper_green` (class 2)

This is a simple heuristic. It works well for bell peppers because they have
strongly saturated colours and the green/red distinction is visually obvious.

### How to run the remapping

```bash
python -m training.remap_classes \
    --src ../fruits-veg-dataset \
    --dst dataset \
    --discard-unknown
```

**Arguments:**

| Flag | Description |
|------|-------------|
| `--src` | Path to the downloaded public dataset root (the folder with `train/`, `val/`, `test/`) |
| `--dst` | Where to write the remapped 14-class dataset. Use `dataset` so it matches `data/yolo_14class.yaml` |
| `--discard-unknown` | Drop any bounding boxes whose source class doesn't map to one of your 14 classes |

### Expected output

```
dataset/
├── train/
│   ├── images/    ← same images (copied or symlinked)
│   └── labels/    ← .txt files with class IDs 0–13
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Verify the remapping

```bash
# Check that class IDs are now 0-13 only
cat dataset/train/labels/*.txt | awk '{print $1}' | sort -n | uniq -c | sort -rn

# Expected output: counts for classes 0 through 13, nothing else
```

You should see roughly balanced counts across the 14 classes. Some imbalance is
normal — the public dataset may have more apple images than peach images.

### Validate with the validation script

```bash
python -m training.validate_dataset --dataset dataset --num-classes 14
```

This checks:
- Every image has a matching label file (and vice versa)
- All class IDs are in the range 0–13
- All bounding box coordinates are in [0, 1]
- Class distribution (flags severe imbalance)

---

## 4. Generate Objectness Labels for Pipeline C

### What objectness means

Pipeline C uses a **two-stage** approach:

1. **Stage 1 — YOLO objectness detector:** Finds *where* objects are, without caring *what* they are. Every bounding box gets the same class label: `0` ("object").
2. **Stage 2 — CNN classifier:** Takes each cropped region and classifies it into one of the 14 classes.

To train the objectness YOLO, you take the 14-class labels you just created and
replace every class ID with `0`. The bounding boxes stay the same — you're just
telling YOLO "there's *something* here" instead of "there's an *apple* here."

### How to run it

```bash
python -m training.prepare_objectness_labels \
    --src dataset \
    --dst dataset_objectness
```

**What this does:**
- Reads every `.txt` label file from `dataset/{train,val,test}/labels/`
- Replaces all class IDs with `0`
- Writes new label files to `dataset_objectness/{train,val,test}/labels/`
- Symlinks (or copies) the images directory so you don't duplicate ~15,000 images

### Expected output

```
dataset_objectness/
├── train/
│   ├── images/    ← symlink to dataset/train/images/
│   └── labels/    ← .txt files with all class IDs = 0
├── val/
│   ├── images/    ← symlink to dataset/val/images/
│   └── labels/
└── test/
    ├── images/    ← symlink to dataset/test/images/
    └── labels/
```

### Verify

```bash
# Every class ID should be 0
cat dataset_objectness/train/labels/*.txt | awk '{print $1}' | sort | uniq -c

# Expected: a single line showing count for class "0"
```

---

## 5. Annotate Your 120 Test Images in Roboflow

Your 120 hand-photographed test images need bounding-box annotations. This
section walks you through doing it in Roboflow (free tier).

### 5.1 Create a Roboflow account

1. Go to [roboflow.com](https://roboflow.com) and sign up (free)
2. You get unlimited annotation on the free plan

### 5.2 Create a new project

1. Click **"Create New Project"**
2. **Project name:** `SnapShelf-Exp2-TestSet`
3. **Project type:** Select **"Object Detection"**
4. **Annotation group:** `SnapShelf-Exp2-TestSet` (default)
5. Click **Create**

### 5.3 Define the 14 class names

In the project settings or when you start annotating, add these **exact** class
names (spelling and underscores matter — they must match `config.py`):

```
apple
banana
bell_pepper_green
bell_pepper_red
carrot
cucumber
grape
lemon
onion
orange
peach
potato
strawberry
tomato
```

### 5.4 Upload your images

1. Click **"Upload"** in your project
2. Drag and drop all 120 `.jpg` files from `dataset_exp2/images/`
3. Wait for the upload to complete
4. Click **"Save and Continue"**

### 5.5 Annotate

For each image:

1. Click on the image to open the annotation editor
2. Draw a **tight bounding box** around every fruit/vegetable item
3. Select the correct class from your 14 labels

**Annotation rules:**

| Scenario | Rule |
|----------|------|
| Grape cluster | Draw **one box** around the entire cluster (not per grape) |
| Strawberries | Draw **one box per strawberry** |
| Partially occluded item | Still annotate it — draw the box around the visible portion |
| Item at image edge | Box it as much as is visible |
| Green vs. red pepper | Use `bell_pepper_green` or `bell_pepper_red` based on the actual colour |
| Unripe tomato (green) | Still use `tomato` — the class is defined by the item, not its colour |

**Tips for speed:**
- Use keyboard shortcuts: `B` for bounding box tool
- After drawing a box, type the first few letters of the class name to filter
- Roboflow auto-suggests the most recently used class

### 5.6 Export in YOLO format

1. Click **"Generate"** → **"Create New Version"**
2. **Preprocessing:** Resize → **640×640** (matches YOLO training size)
3. **Augmentation:** None (this is a test set — no augmentation)
4. Click **"Generate"**
5. Once generated, click **"Export Dataset"**
6. Format: select **"YOLOv8"**
7. Download the `.zip` file

### 5.7 Place files in the project

Extract the downloaded `.zip`. You will get a folder like:

```
SnapShelf-Exp2-TestSet-1/
├── test/
│   ├── images/    ← your 120 images (possibly resized)
│   └── labels/    ← 120 .txt annotation files
└── data.yaml
```

Copy **only the label files** into your project:

```bash
# Copy label files
cp SnapShelf-Exp2-TestSet-1/test/labels/*.txt dataset_exp2/labels/
```

> **Important:** Keep your original full-resolution images in
> `dataset_exp2/images/`. Do not replace them with the resized Roboflow copies.
> YOLO resizes internally during inference — your images should stay at their
> original resolution.

### 5.8 Verify: 120 .txt files matching 120 .jpg files

```bash
# Count images and labels
ls dataset_exp2/images/*.jpg | wc -l    # Should be 120
ls dataset_exp2/labels/*.txt | wc -l    # Should be 120

# Check that every image has a matching label
for img in dataset_exp2/images/*.jpg; do
    stem=$(basename "$img" .jpg)
    if [ ! -f "dataset_exp2/labels/${stem}.txt" ]; then
        echo "MISSING label for: $img"
    fi
done

# Check that every label has a matching image
for lbl in dataset_exp2/labels/*.txt; do
    stem=$(basename "$lbl" .txt)
    if [ ! -f "dataset_exp2/images/${stem}.jpg" ]; then
        echo "MISSING image for: $lbl"
    fi
done
```

Also verify class IDs are in range:

```bash
cat dataset_exp2/labels/*.txt | awk '{print $1}' | sort -n | uniq -c | sort -rn
# All class IDs should be 0-13
```

---

## 6. Train the YOLO Models

You need to train two YOLO models:

1. **14-class YOLO** (Pipeline B) — detects and classifies into 14 fruit/veg classes
2. **Objectness YOLO** (Pipeline C) — detects "something is here" (1 class)

### 6.1 Pre-training checks

**Check GPU availability:**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Training on GPU takes **1–3 hours**. On CPU it can take **12–24 hours**. A GPU
is strongly recommended.

**Check that dataset paths match the YAML configs:**

The file `data/yolo_14class.yaml` expects:
```yaml
path: ../dataset
train: train/images
val: val/images
```

This means the training images should be at `dataset/train/images/` relative to
the project root. Verify:

```bash
ls dataset/train/images/ | head -5    # Should show .jpg files
ls dataset/train/labels/ | head -5    # Should show .txt files
```

Similarly, `data/yolo_objectness.yaml` expects `../dataset_objectness`.

**If your dataset is in a different location,** regenerate the YAML:

```bash
python -m training.data_yaml_generator \
    --dataset /path/to/your/14class/dataset \
    --dataset-obj /path/to/your/objectness/dataset
```

### 6.2 Train the 14-class YOLO (Pipeline B)

```bash
python main.py train yolo-14
```

Or with custom settings:

```bash
python main.py train yolo-14 --epochs 100 --batch 16
```

**Default hyperparameters** (from `config.py`):

| Setting | Value |
|---------|-------|
| Base model | YOLOv8s (pre-trained on COCO) |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640 |
| Learning rate | 0.01 |
| Early stopping patience | 15 epochs |
| Random seed | 42 |

**What to expect during training:**

- Ultralytics will download `yolov8s.pt` (~22 MB) on first run
- You will see a progress bar for each epoch
- Training logs are saved to `runs/yolo_14class/train/`
- Metrics (mAP, loss) are printed per epoch
- Early stopping will halt training if validation loss stops improving for 15 epochs
- The best weights are automatically copied to `weights/yolo_14class_best.pt`

### 6.3 Train the objectness YOLO (Pipeline C)

```bash
python main.py train yolo-obj
```

Or with custom settings:

```bash
python main.py train yolo-obj --epochs 100 --batch 16
```

This trains identically to the 14-class model but with `nc=1` (one class:
"object"). It should converge faster since the task is simpler.

### 6.4 Verify training succeeded

```bash
# Check that weight files exist
ls -la weights/yolo_14class_best.pt
ls -la weights/yolo_objectness_best.pt

# Run a smoke test on a single image
python main.py yolo-14 dataset_exp2/images/IMG_001.jpg
python main.py yolo-cnn dataset_exp2/images/IMG_001.jpg
```

Both commands should print a JSON result with an `inventory` field. If you see
an error about missing weights, training did not complete successfully — check
the logs in `runs/`.

You can also inspect training curves in TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## 7. Set Up Pipeline C's CNN (EfficientNet from Experiment 1)

Pipeline C uses an EfficientNet-B0 classifier (from Experiment 1) to classify
the crops extracted by the objectness YOLO. You need to copy the trained weights.

### 7.1 Copy the winning weights

The CNN from Experiment 1 should be saved as a `.pth` file. Copy it to:

```bash
cp /path/to/experiment1/best_efficientnet.pth weights/cnn_winner.pth
```

### 7.2 Verify config.py settings

Open `config.py` and confirm these settings match your setup:

```python
cnn_model_name: str = "efficientnet"   # Must match the architecture you trained
cnn_weights: str = "weights/cnn_winner.pth"
cnn_img_size: int = 224                # EfficientNet-B0 default
cnn_crop_padding: float = 0.10         # 10% context padding around crops
```

### 7.3 Quick verification

```bash
# Verify the file exists and has reasonable size (~20 MB for EfficientNet-B0)
ls -lh weights/cnn_winner.pth

# Run a smoke test
python main.py yolo-cnn dataset_exp2/images/IMG_001.jpg
```

The output should show an inventory with classified items.

### Important note about domain gap

The CNN was trained on **clean, single-item, centred images** (Experiment 1).
In Pipeline C, it receives **YOLO-cropped regions** from multi-item real scenes.
These crops may include:

- Partial objects (crop cuts off part of the item)
- Background clutter (nearby items visible)
- Different lighting conditions
- Items at unusual angles

This **domain gap** is expected and is a legitimate finding for your
dissertation. It reveals a real limitation of the detect-then-classify approach.
Do not try to fix it — document it.

---

## 8. Generate Degraded Image Variants

To test robustness, you create three degraded versions of your 120 test images.
The labels stay the same — degradation does not move or change the objects.

### 8.1 The three degradations

| ID | Degradation | Parameters | What it simulates |
|:--:|-------------|-----------|-------------------|
| D1 | **Gaussian blur** | kernel=7, sigma=3.0 | Out-of-focus camera or shaky hands |
| D2 | **Gaussian noise** | mean=0, sigma=25 | Low-light sensor noise (grainy photo) |
| D3 | **JPEG compression** | quality=15 | Images sent through messaging apps or cheap uploads |

These cover the three main sources of quality loss in a mobile app:
**optical** (blur), **sensor** (noise), and **compression** (JPEG artifacts).

### 8.2 Run the degradation script

```bash
python -m evaluation.generate_degradations \
    --src dataset_exp2/images \
    --dst dataset_exp2
```

### 8.3 Expected output

```
dataset_exp2/
├── images/              ← 120 clean originals (unchanged)
├── labels/              ← 120 .txt files (shared by all variants)
├── images_d1_blur/      ← 120 blurred images
├── images_d2_noise/     ← 120 noisy images
└── images_d3_jpeg/      ← 120 JPEG-compressed images
```

### Verify

```bash
# Each folder should have 120 images
ls dataset_exp2/images_d1_blur/*.jpg | wc -l    # 120
ls dataset_exp2/images_d2_noise/*.jpg | wc -l   # 120
ls dataset_exp2/images_d3_jpeg/*.jpg | wc -l    # 120
```

Open a few images visually to confirm the degradations are visible but items
are still recognisable by a human:
- **D1 (blur):** Image should look like an out-of-focus photo
- **D2 (noise):** Image should look grainy, like a photo taken in dim light
- **D3 (JPEG):** Image should show blocky compression artifacts

---

## 9. Run All 12 Evaluation Runs

### 9.1 The 12-run matrix

3 pipelines × 4 image conditions = **12 runs**.

| Run | Pipeline | Images Folder | Labels Folder |
|:---:|----------|--------------|--------------|
| 1 | A (VLM) | `dataset_exp2/images` | `dataset_exp2/labels` |
| 2 | A (VLM) | `dataset_exp2/images_d1_blur` | `dataset_exp2/labels` |
| 3 | A (VLM) | `dataset_exp2/images_d2_noise` | `dataset_exp2/labels` |
| 4 | A (VLM) | `dataset_exp2/images_d3_jpeg` | `dataset_exp2/labels` |
| 5 | B (YOLO) | `dataset_exp2/images` | `dataset_exp2/labels` |
| 6 | B (YOLO) | `dataset_exp2/images_d1_blur` | `dataset_exp2/labels` |
| 7 | B (YOLO) | `dataset_exp2/images_d2_noise` | `dataset_exp2/labels` |
| 8 | B (YOLO) | `dataset_exp2/images_d3_jpeg` | `dataset_exp2/labels` |
| 9 | C (YOLO+CNN) | `dataset_exp2/images` | `dataset_exp2/labels` |
| 10 | C (YOLO+CNN) | `dataset_exp2/images_d1_blur` | `dataset_exp2/labels` |
| 11 | C (YOLO+CNN) | `dataset_exp2/images_d2_noise` | `dataset_exp2/labels` |
| 12 | C (YOLO+CNN) | `dataset_exp2/images_d3_jpeg` | `dataset_exp2/labels` |

Note: the `labels/` folder is **always the same** — degradation does not change
the ground truth annotations.

### 9.2 Pre-flight checks

Before running, verify everything is in place:

```bash
# 1. Weights exist
ls weights/yolo_14class_best.pt
ls weights/yolo_objectness_best.pt
ls weights/cnn_winner.pth

# 2. OpenAI API key is set (for Pipeline A)
echo $OPENAI_API_KEY    # Should print your key (or set it in .env)

# 3. Test images and labels exist
ls dataset_exp2/images/*.jpg | wc -l         # 120
ls dataset_exp2/labels/*.txt | wc -l         # 120
ls dataset_exp2/images_d1_blur/*.jpg | wc -l # 120
ls dataset_exp2/images_d2_noise/*.jpg | wc -l # 120
ls dataset_exp2/images_d3_jpeg/*.jpg | wc -l  # 120

# 4. Environment validation
python main.py --validate
```

### 9.3 Run the evaluations

**Run 1–4: All pipelines on clean images**

```bash
python main.py evaluate \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels \
    --output results/clean
```

This runs all three pipelines (VLM, YOLO-14, YOLO+CNN) on the clean test set
and saves results to `results/clean/`.

**Run 5–8: All pipelines on D1 (blur)**

```bash
python main.py evaluate \
    --images dataset_exp2/images_d1_blur \
    --labels dataset_exp2/labels \
    --output results/d1_blur
```

**Run 9–12: D2 (noise) and D3 (JPEG)**

```bash
python main.py evaluate \
    --images dataset_exp2/images_d2_noise \
    --labels dataset_exp2/labels \
    --output results/d2_noise

python main.py evaluate \
    --images dataset_exp2/images_d3_jpeg \
    --labels dataset_exp2/labels \
    --output results/d3_jpeg
```

**To run a single pipeline only** (e.g. just VLM on clean):

```bash
python main.py evaluate \
    --images dataset_exp2/images \
    --labels dataset_exp2/labels \
    --pipelines vlm \
    --output results/clean_vlm_only
```

### 9.4 What output to expect

For each run, you will see:

1. **Console progress:** `[1/120] IMG_001.jpg... 342ms` for each image
2. **JSON predictions:** `results/<condition>/<pipeline>_predictions.json`
3. **Full results:** `results/<condition>/<pipeline>_results.json`
4. **Comparison summary:** `results/<condition>/comparison_summary.json`
5. **Confusion matrices:** `results/<condition>/<pipeline>_confusion.png`
6. **Bar chart:** `results/<condition>/comparison_bars.png`
7. **LaTeX table:** `results/<condition>/comparison_table.tex`

### 9.5 Expected timing

| Pipeline | Approx. time per image | Total (120 images) |
|----------|----------------------|-------------------|
| A (VLM) | 1–3 seconds (API call) | ~3–6 minutes |
| B (YOLO) | 20–80 ms (GPU) / 200–500 ms (CPU) | ~5 sec – 1 min |
| C (YOLO+CNN) | 30–120 ms (GPU) / 300–800 ms (CPU) | ~6 sec – 1.5 min |

Total for all 12 runs: **~30–45 minutes** (mostly VLM API time).

### 9.6 Verify results

After all four evaluation runs complete:

```bash
# Check that result files exist for all conditions
ls results/clean/comparison_summary.json
ls results/d1_blur/comparison_summary.json
ls results/d2_noise/comparison_summary.json
ls results/d3_jpeg/comparison_summary.json

# Quick-check F1 scores
python -c "
import json
for cond in ['clean', 'd1_blur', 'd2_noise', 'd3_jpeg']:
    with open(f'results/{cond}/comparison_summary.json') as f:
        summary = json.load(f)
    print(f'\n--- {cond} ---')
    for pipe, data in summary.items():
        f1 = data['metrics']['f1']
        print(f'  {pipe}: F1 = {f1:.4f}')
"
```

---

## 10. Dissertation Methodology Text

Below are ready-to-adapt paragraphs you can use in your dissertation. Edit them
to match your institution's style and requirements.

### 10.1 Experimental design

> *"Experiment 2 evaluates three end-to-end pipelines on an identical test set
> of 120 photographs, each containing 2–8 items from a 14-class fruit and
> vegetable taxonomy. Pipeline A uses a vision-language model (GPT-5.2,
> OpenAI) with a constrained prompt that restricts output to the 14 target
> classes. Pipeline B uses a YOLOv8s object detector fine-tuned to directly
> predict all 14 classes. Pipeline C uses a two-stage approach: a YOLOv8s model
> trained as a class-agnostic objectness detector to localise items, followed by
> an EfficientNet-B0 classifier (the winning model from Experiment 1) to
> classify each cropped region."*

### 10.2 Training/test separation

> *"To ensure a fair comparison, training and test data were strictly separated.
> Pipelines B and C were fine-tuned on the publicly available
> Fruits-And-Vegetables-Detection-Dataset (Kaggle, ~15,000 images, 63 classes),
> remapped to the 14-class taxonomy used in this study. The test set comprised
> 120 original photographs taken by the author across five real-world settings
> (kitchen counter, refrigerator shelf, dining table, grocery bag, and chopping
> board), balanced across three difficulty tiers and three camera angles. These
> test images were never used during training."*

### 10.3 Fairness table

> **Table X — Data usage per pipeline**
>
> | | Training Data | Test Data |
> |---|---|---|
> | Pipeline A (VLM) | Pre-trained by OpenAI (not fine-tuned) | 120 original photographs |
> | Pipeline B (YOLO-14) | Public dataset, remapped to 14 classes | 120 original photographs |
> | Pipeline C (YOLO+CNN) | YOLO: public dataset (objectness labels); CNN: Experiment 1 single-item images | 120 original photographs |

### 10.4 Robustness evaluation

> *"To assess robustness, three image degradations were applied to the test set:
> Gaussian blur (kernel=7, sigma=3.0, simulating an out-of-focus camera),
> additive Gaussian noise (sigma=25, simulating low-light sensor noise), and
> JPEG compression at quality level 15 (simulating lossy transmission through
> messaging applications). These degradations represent the three principal
> sources of image quality loss in a mobile application workflow: optical,
> sensor, and compression artefacts. Each pipeline was evaluated on all four
> conditions (clean plus three degraded), yielding 12 evaluation runs. Ground
> truth annotations were shared across all conditions, as degradation does not
> alter object locations."*

### 10.5 Metrics

> *"Performance was assessed using count-based micro-averaged precision, recall,
> and F1 score. For each image, true positives were computed as the minimum of
> predicted and actual count per class, false positives as the excess of
> predicted over actual count, and false negatives as the shortfall. Metrics
> were aggregated across all images and classes to produce micro-averaged scores.
> Robustness was measured as the percentage F1 drop from the clean baseline to
> each degraded condition. Statistical significance between pipeline pairs was
> assessed using the Wilcoxon signed-rank test with Bonferroni correction for
> multiple comparisons (adjusted alpha = 0.004)."*

### 10.6 Limitations (for the Discussion chapter)

> *"Several limitations should be acknowledged. First, the test set of 120
> images, while purpose-built to reflect the target deployment environment with
> controlled variation in difficulty, setting, and angle, is small by industry
> standards. Second, all images were captured using a single device by a single
> photographer, which controls for device variance but limits generalisability
> to other camera hardware. Third, Pipeline C's CNN was trained on clean,
> single-item, centred images (Experiment 1), introducing a domain gap when
> classifying YOLO-cropped regions from cluttered multi-item scenes. This is
> reported as a finding rather than a flaw, as it reveals a genuine limitation
> of the detect-then-classify paradigm. Fourth, while Pipeline A (GPT-5.2)
> was queried with temperature=0 for reproducibility, the VLM's outputs are not
> guaranteed to be fully deterministic across API versions. Finally, results are
> specific to the 14-class taxonomy and may not generalise to larger class sets."*

---

## Quick Reference — File Locations

| File / Folder | Purpose |
|--------------|---------|
| `config.py` | All 14 classes, frozen hyperparameters |
| `main.py` | CLI entry point (`vlm`, `yolo-14`, `yolo-cnn`, `evaluate`, `train`) |
| `data/yolo_14class.yaml` | YOLO training config (14 classes, nc=14) |
| `data/yolo_objectness.yaml` | YOLO training config (1 class, nc=1) |
| `dataset/` | Remapped 14-class training data (from public dataset) |
| `dataset_objectness/` | Objectness training data (class IDs all = 0) |
| `dataset_exp2/images/` | Your 120 test images |
| `dataset_exp2/labels/` | Your 120 YOLO annotation files |
| `dataset_exp2/images_d1_blur/` | Blurred test images |
| `dataset_exp2/images_d2_noise/` | Noisy test images |
| `dataset_exp2/images_d3_jpeg/` | JPEG-compressed test images |
| `weights/yolo_14class_best.pt` | Trained 14-class YOLO weights |
| `weights/yolo_objectness_best.pt` | Trained objectness YOLO weights |
| `weights/cnn_winner.pth` | EfficientNet-B0 from Experiment 1 |
| `results/` | Evaluation outputs (JSON, PNG, LaTeX) |
| `logs/` | Structured JSONL experiment logs |
| `training/remap_classes.py` | Remap 63-class public dataset → 14 classes |
| `training/prepare_objectness_labels.py` | Remap 14-class labels → objectness (class 0) |
| `training/train_yolo_14class.py` | Train Pipeline B's YOLO |
| `training/train_yolo_objectness.py` | Train Pipeline C's objectness YOLO |
| `training/validate_dataset.py` | Validate dataset integrity |
| `training/data_yaml_generator.py` | Generate YOLO data.yaml files |
