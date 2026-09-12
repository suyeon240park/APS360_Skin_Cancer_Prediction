# APS360 Skin Lesion Classification

This University of Toronto APS360 project explores **four-class skin-lesion image classification** with transfer learning using a ResNet50 backbone and PyTorch. The project classes are:

- basal cell carcinoma (`bcc`)
- benign lesions (`benign`)
- melanoma (`melanoma`)
- squamous cell carcinoma (`scc`)

> **Educational project only:** this repository is a machine-learning coursework project, not diagnostic software or a substitute for medical evaluation.

## Repository Structure

- `Baseline.ipynb` — baseline experiments from the original coursework.
- `final_model.ipynb` — original training/analysis notebook preserved as a record of the project work.
- `model.py` — corrected reusable ResNet50 model definition and preprocessing pipeline.
- `test_model_one_sample.py` — command-line inference for one image.
- `test_model.py` — command-line evaluation for a labeled `ImageFolder` dataset.

## Canonical Model Code

For reusable inference, use `model.py` rather than copying the model class from the historical notebooks.

Torchvision's ResNet forward pass already performs global average pooling and flattening before its final fully connected layer. After replacing ResNet50's `fc` layer with `nn.Identity()`, the backbone therefore returns a feature vector directly. The reusable implementation in `model.py` correctly sends that vector to the classifier without applying a second `AdaptiveAvgPool2d` operation.

The classifier returns **logits**. Softmax is applied only when probabilities are needed during inference, which keeps the model suitable for standard classification losses during training.

## Setup

Create a virtual environment and install the inference dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Model Checkpoint

The trained checkpoint is not committed to this repository. To run inference, provide a compatible PyTorch `state_dict` checkpoint such as:

```text
best_model.pth
```

The loader also accepts a checkpoint dictionary containing a `state_dict` entry.

## Single-Image Inference

```bash
python test_model_one_sample.py path/to/image.jpg --model best_model.pth
```

Example output:

```text
Predicted class: melanoma
Confidence: 82.14%
```

The confidence value is the model's softmax probability for the predicted class. It should not be interpreted as a calibrated medical probability.

## Evaluate a Labeled Dataset

`test_model.py` expects the standard Torchvision `ImageFolder` directory structure:

```text
test_data/
├── bcc/
├── benign/
├── melanoma/
└── scc/
```

Run:

```bash
python test_model.py test_data --model best_model.pth
```

Optional arguments include:

```bash
python test_model.py test_data \
  --model best_model.pth \
  --batch-size 32 \
  --device cpu
```

If CUDA is available, the script selects it by default.

## Preprocessing

The reusable inference pipeline applies:

1. resize to `224 x 224`;
2. conversion to a PyTorch tensor;
3. ImageNet normalization using the standard ResNet mean and standard deviation.

## Notes on the Original Notebooks

The notebooks are retained as historical coursework artifacts and include the original experimentation, training, and analysis. The standalone Python modules were added to provide a smaller, clearer, and reproducible inference path for the public repository. Where the notebook implementation and `model.py` differ, **`model.py` is the intended source of truth for inference**.

## Limitations

- Skin-lesion datasets can contain class imbalance, acquisition bias, demographic bias, and dataset-specific artifacts.
- Performance on a held-out course dataset does not establish clinical validity or real-world generalization.
- Softmax confidence is not equivalent to diagnostic confidence.
- This repository does not include a production inference service, clinical validation, or regulatory evaluation.
