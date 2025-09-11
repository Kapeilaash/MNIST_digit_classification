<div align="center">

# MNIST Digit Classification with a Neural Network

Handwritten digit recognition on the classic MNIST dataset using a simple (yet extendable) neural network pipeline.

![Workflow](workflow.png)

</div>

## 1. Overview
This repository aims to provide a clean, reproducible baseline for classifying handwritten digits (0–9) from the MNIST dataset. It includes (or is prepared to include) scripts for: data download, preprocessing, model definition, training, evaluation, and inference on custom images. The goal is to keep the core logic simple while making it easy to extend (deeper networks, CNNs, augmentation, experiment tracking, etc.).

## 2. Project Structure (Planned / Recommended)
If you have not yet created these folders, you can do so—this layout is a suggested convention:

```
MNIST_digit_classification/
├─ data/
│  ├─ raw/                # Original downloaded MNIST
│  └─ processed/          # Cached tensors / normalized data
├─ notebooks/             # Jupyter notebooks for exploration
├─ src/
│  ├─ data/
│  │  ├─ download_mnist.py
│  │  └─ preprocess.py
│  ├─ models/
│  │  ├─ network.py       # Model architecture (MLP / CNN)
│  │  ├─ train.py         # Training loop
│  │  ├─ evaluate.py      # Metrics & evaluation script
│  │  └─ predict.py       # Inference on a single image / folder
│  └─ utils/
│     └─ seed.py          # Reproducibility helpers
├─ models/                # Saved model weights (.pth)
├─ images/                # Figures (confusion matrix, samples)
├─ workflow.png           # Workflow diagram (ensure this file exists)
├─ requirements.txt       # Python dependencies (to be added)
└─ README.md
```

> NOTE: At the moment only `README.md` might exist. Add the rest incrementally as you implement each component.

## 3. Workflow Explanation
1. **Dataset Acquisition** – Download MNIST (60k train / 10k test). Libraries like `torchvision.datasets.MNIST` or `tensorflow.keras.datasets.mnist` simplify this.
2. **Image Processing** – Normalize pixel values (e.g., mean=0.1307, std=0.3081 for PyTorch) and optionally apply augmentation (for CNN variants).
3. **Train / Test Split** – Standard MNIST split (optionally carve out a validation set from training, e.g., 55k train / 5k val).
4. **Neural Network Training** – Start with a simple MLP or small CNN. Optimize cross-entropy loss with Adam/SGD.
5. **Evaluation** – Report accuracy, confusion matrix, per-class accuracy.
6. **Inference** – Load trained weights and predict digits for new handwritten images (after resizing & normalization).

## 4. Requirements (Example)
A minimal `requirements.txt` (create later) might include:
```
torch
torchvision
numpy
matplotlib
scikit-learn
Pillow
```
Optional extras: `tqdm`, `rich`, `tensorboard`, `wandb`.

## 5. Environment Setup (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt   # (after you create it)
```

If you have not created `requirements.txt` yet, install packages manually:
```powershell
pip install torch torchvision numpy matplotlib scikit-learn Pillow
```

## 6. Training Script (Planned Usage)
Once `train.py` exists (example interface):
```powershell
python -m src.models.train --epochs 10 --batch-size 128 --lr 0.001 --model mlp --save-path models/mnist_mlp.pth
```
Suggested CLI arguments:
* `--model` (mlp|cnn)
* `--epochs`
* `--batch-size`
* `--lr`
* `--seed`
* `--device` (cpu|cuda)
* `--save-path`

## 7. Evaluation
```powershell
python -m src.models.evaluate --weights models/mnist_mlp.pth --batch-size 512 --device cpu
```
Outputs: overall accuracy, per-class accuracy, confusion matrix (save to `images/`).

## 8. Inference on a Custom Image
Prepare a 28x28 grayscale image (or an arbitrary handwritten digit image that you preprocess). Example planned usage:
```powershell
python -m src.models.predict --weights models/mnist_mlp.pth --image path/to/digit.png
```
Returns predicted digit + probabilities.

## 9. Reproducibility
* Set seeds: `torch.manual_seed(seed)` and `numpy.random.seed(seed)`.
* Disable nondeterministic CUDA (optional):
	```python
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	```

## 10. Metrics & Baselines
| Model | Params | Epochs | Test Accuracy |
|-------|--------|--------|---------------|
| (todo) MLP (hidden=[256,128]) | ~ | 10 | ~97% |
| (todo) Simple CNN | ~ | 10 | ~99% |

Populate this table as you run experiments.

## 11. Experiment Log Template
```
Date: 2025-08-20
Model: MLP (256-128)
Optimizer: Adam (lr=1e-3)
Batch size: 128
Epochs: 10
Best Val Acc: 97.3%
Test Acc: 97.1%
Notes: Overfitting minimal, might add dropout 0.2.
```

## 12. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Image not shown in README | `workflow.png` missing or wrong path | Ensure file name matches & pushed to repo |
| Import errors | Modules not found | Activate venv, reinstall requirements |
| Low accuracy (<90%) | Incorrect normalization / label mismatch | Verify transforms & dataset download |

## 13. Roadmap
- [ ] Add `requirements.txt`
- [ ] Implement `src/data/download_mnist.py`
- [ ] Implement baseline MLP (`src/models/network.py`)
- [ ] Add training / evaluation scripts
- [ ] Add CNN variant
- [ ] Add inference script + sample images
- [ ] Add confusion matrix & sample predictions to `images/`
- [ ] Integrate experiment tracking (TensorBoard / Weights & Biases)

## 14. Contributing
Pull requests are welcome. For major changes, open an issue first to discuss what you’d like to change.
---
Happy experimenting! Feel free to open issues for questions or feature suggestions.


