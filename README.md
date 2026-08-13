# ♻️ AI Agent for Garbage Classification and Smart Recycling Recommendation

A final-year B.Tech project that classifies waste images with deep learning
(EfficientNet-B3, ResNet50, MobileNetV3 — transfer learning) and then
generates a structured recycling/disposal recommendation for the predicted
category, served through a Streamlit web app.

## Problem Statement

Improper waste segregation is one of the biggest bottlenecks in effective
recycling. Most people don't reliably know which bin an item belongs in,
and manual sorting at scale is slow and error-prone. An automated system
that can look at a photo of an item and both (a) classify its material type
and (b) tell the user exactly how to dispose of or reuse it, can meaningfully
reduce contamination in recycling streams and improve household waste
segregation habits.

## Objectives

- Classify a waste image into its material category using a CNN trained
  via transfer learning.
- Compare three architectures (EfficientNet-B3, ResNet50, MobileNetV3-Large)
  fairly on the same data/splits/training regime and pick the best one on
  measured performance, not by assumption.
- Turn the predicted category into a concrete, actionable recycling/reuse
  recommendation, mapped from a structured knowledge base.
- Provide model explainability (Grad-CAM) so predictions aren't a black box.
- Ship it as a usable Streamlit web application.

## Dataset

[Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
(Kaggle, uploader: asdasdasasdas). The notebook auto-detects the actual
class folders and counts at run time rather than assuming them — see
`src/dataset.py::discover_class_folders`.

## Technologies

Python · PyTorch · Torchvision · EfficientNet-B3 · ResNet50 · MobileNetV3 ·
scikit-learn · OpenCV · Streamlit · Matplotlib/Seaborn

## Architecture

```
        USER
          |
          v
  Upload Garbage Image
          |
          v
  Image Preprocessing (resize, normalize)
          |
          v
  +---------------------------+
  |   Deep Learning Models    |
  |  EfficientNet-B3          |
  |  ResNet50                 |
  |  MobileNetV3              |
  +---------------------------+
          |
          v
   Model Comparison (Section 11)
          |
          v
      Best Model
          |
          v
  Garbage Classification (Top-1, Top-3, confidence)
          |
          v
  Confidence Threshold Check
          |
          v
  Recycling Knowledge Base (src/recycling.py)
          |
          v
  Smart Recommendation (disposal, reuse, impact, safety)
          |
          v
        USER (Streamlit UI)
```

Everything above the "Streamlit UI" line lives in `src/`, independent of
the web framework — the CNN model always determines the class; an optional
LLM (not required to run the project) could only rephrase the structured
recommendation, never override the classification (see Section 21 of the
original spec, and `src/recycling.py`'s docstring).

## Project Structure

```
garbage-classification-ai/
├── data/{raw,processed,splits}/
├── notebooks/garbage_classification_colab.ipynb
├── src/
│   ├── config.py        # all hyperparameters & paths
│   ├── dataset.py        # discovery, cleaning, stratified split
│   ├── preprocessing.py  # transforms, imbalance-aware DataLoaders
│   ├── models.py          # EfficientNet-B3 / ResNet50 / MobileNetV3 builders
│   ├── train.py           # reusable AMP training loop
│   ├── evaluate.py        # metrics, comparison table, best-model selection
│   ├── predict.py         # single-image inference pipeline
│   ├── recycling.py       # recycling knowledge base + recommendation logic
│   ├── gradcam.py         # Grad-CAM explainability
│   └── utils.py           # seeding, checkpoints, plotting helpers
├── models/                # trained checkpoints (.pth) — gitignored
├── results/{figures,metrics}/, model_comparison.csv
├── app/streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
git clone <your-repo-url>
cd garbage-classification-ai
pip install -r requirements.txt
```

## Dataset Setup (Kaggle API)

1. Go to your Kaggle account → **Settings → API → Create New Token**. This
   downloads `kaggle.json`.
2. In Colab, upload it and run:
   ```python
   from google.colab import files
   files.upload()  # select kaggle.json

   import os
   os.makedirs("/root/.kaggle", exist_ok=True)
   os.system("cp kaggle.json /root/.kaggle/")
   os.system("chmod 600 /root/.kaggle/kaggle.json")
   ```
3. Download and extract:
   ```bash
   kaggle datasets download -d asdasdasasdas/garbage-classification -p data/raw --unzip
   ```

**Manual alternative:** download the zip directly from the
[Kaggle dataset page](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification),
then upload/unzip it into `data/raw/` (in Colab: use the file browser's
upload button, then `!unzip <file>.zip -d data/raw`).

## Training

Open `notebooks/garbage_classification_colab.ipynb` in Google Colab
(**Runtime → Change runtime type → GPU**), and run the cells top to bottom.
It downloads the data, explores/cleans it, splits it, trains all three
models, evaluates them, builds the comparison table, selects the best
model, and saves it to `models/best_model.pth`.

Alternatively, from the `src/` modules directly:

```python
from src import config, dataset, preprocessing, models, train, utils
utils.set_seed()

class_to_images = dataset.discover_class_folders(config.RAW_DATA_DIR)
report = dataset.audit_dataset(class_to_images)
clean = dataset.clean_dataset(class_to_images, report)
train_items, val_items, test_items, class_names = dataset.stratified_split(clean)
train_loader, val_loader, test_loader = preprocessing.build_dataloaders(
    train_items, val_items, test_items, class_names
)

model = models.build_model("resnet50", num_classes=len(class_names))
criterion = train.make_criterion()
optimizer = train.make_optimizer(model)
scheduler = train.make_scheduler(optimizer)
model, history, best_val_acc = train.train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs=config.NUM_EPOCHS_HEAD, model_name="resnet50",
    checkpoint_path=config.MODEL_CHECKPOINT_PATHS["resnet50"], class_names=class_names,
)
```

## Evaluation

`src/evaluate.py` computes accuracy, weighted precision/recall/F1, macro-F1,
a full classification report, confusion matrix, most-confused class pairs,
inference latency, and model size — then assembles the cross-model
comparison table (`results/model_comparison.csv`) and picks the best model
by a combined F1 + inference-speed score (see `evaluate.select_best_model`).

## Application

```bash
streamlit run app/streamlit_app.py
```

Upload a JPG/JPEG/PNG image to see the predicted category, confidence,
top-3 predictions, recycling recommendation, and (optionally) a Grad-CAM
heatmap explaining which regions drove the prediction.

## Example

```
Detected Waste: Plastic
Confidence: 96.3%

Recyclability: Recyclable

Recommended Action:
Check the resin code... most curbside programs accept #1 and #2 plastics.

Steps:
1. Empty and rinse the container.
2. Remove non-plastic parts if separable.
3. Separate the cap if your local program requires it.
4. Place in the appropriate recyclable-waste bin per your local guidelines.

Reuse Ideas:
- Plant container
- Storage container
- DIY craft

Environmental Impact:
Plastic waste can persist in the environment for hundreds of years...
```

## Future Enhancements

- Mobile deployment (TensorFlow Lite / ONNX export of MobileNetV3)
- Edge AI on Raspberry Pi / Jetson Nano for smart bins
- IoT-connected smart bins with automatic sorting
- Waste quantity/volume estimation from images
- Real-time camera-based classification (video stream)
- Location-aware recycling centre lookup
- Multilingual recycling recommendations
- Carbon-footprint estimation per disposed item

## Model Storage Note

Trained `.pth` checkpoints are excluded from git (see `.gitignore`) because
they're large binaries unsuited to a normal git repo. Use **Git LFS** or an
external store (Google Drive, Hugging Face Hub, S3) and document the
download link/instructions in your submission if you need to share weights.
