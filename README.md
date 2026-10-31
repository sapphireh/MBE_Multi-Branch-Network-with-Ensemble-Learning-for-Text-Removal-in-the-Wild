# MBE: Multi-Branch Network with Ensemble Learning for Text Removal in the Wild

Research code accompanying our **ACCV 2022** paper.

**Yujie Hou, Jiwei Chen, and Zengfu Wang**

[Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Hou_Multi-Branch_Network_with_Ensemble_Learning_for_Text_Removal_in_the_ACCV_2022_paper.html) · [PDF](https://openaccess.thecvf.com/content/ACCV2022/papers/Hou_Multi-Branch_Network_with_Ensemble_Learning_for_Text_Removal_in_the_ACCV_2022_paper.pdf) · [Author homepage](https://sapphireh.github.io/)

## Overview

Scene text removal replaces text in natural images with plausible background content. MBE combines complementary branches to remove text while preserving background texture:

- A main branch maintains high-resolution texture information.
- Two sub-branches learn multi-scale semantic features from a hierarchy of image patches.
- Feature-level and image-level ensemble fusion combine complementary representations and predictions.
- A patch attention module helps the network focus on text regions.

![Original Figure 3 from the ACCV 2022 paper: the MBE architecture](https://sapphireh.github.io/assets/papers/mbe-figure-3.png)

*Figure 3 from the original paper, hosted on the author's homepage. The paper reports experiments on SCUT-EnsText and SCUT-Syn; see the paper for the experimental protocol and results.*

## Repository contents

| File | Purpose |
| --- | --- |
| [`MBE/HiCarNet.py`](MBE/HiCarNet.py) | Model components and the `HicarNet` class. |
| [`MBE/dataset_RGB.py`](MBE/dataset_RGB.py) | Training, validation, and test dataset classes. |
| [`MBE/data_RGB.py`](MBE/data_RGB.py) | Dataset factory functions. |
| [`MBE/config.py`](MBE/config.py) | YACS configuration defaults and YAML loading. |
| [`MBE/losses.py`](MBE/losses.py) | Charbonnier, edge, reconstruction, perceptual, style, and mask-related losses. |
| [`MBE/utils/`](MBE/utils/) | Image, checkpoint, augmentation, and directory helpers. |
| [`MBE/evaluate_PSNR_SSIM.m`](MBE/evaluate_PSNR_SSIM.m) | MATLAB evaluation of restored images against ground truth. |
| [`MBE/detect.py`](MBE/detect.py) | Text-detection helper requiring an external `lib` package. |

### Scope of this code snapshot

The repository contains model and utility code, but does **not** currently include a complete training launcher, `training.yml`, pretrained model weights, or datasets. The root `main.py` is an IDE example. The root `demo.py` is a legacy MPRNet restoration demo that refers to task directories absent from this repository; it is not an MBE inference entry point.

The checked-in model is named `HicarNet`. Its final image fusion sums the outputs of the `sum1`, `sum2`, and `sum3` attention blocks. The paper describes a ConvLSTM image-fusion module, which is not present in this snapshot. Consult the paper for the full published architecture; this checkout alone does not provide an end-to-end reproduction of the reported results.

## Getting started

Clone the repository and work from its root:

```bash
git clone https://github.com/sapphireh/MBE_Multi-Branch-Network-with-Ensemble-Learning-for-Text-Removal-in-the-Wild.git
cd MBE_Multi-Branch-Network-with-Ensemble-Learning-for-Text-Removal-in-the-Wild
```

Use an isolated Python environment. Install a compatible pair of **PyTorch** and **torchvision** using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) for your Python version and compute platform. Then install the other packages imported by the model:

```bash
python -m pip install numpy pillow opencv-python
```

Configuration and detection/data utilities additionally import `yacs` and `matplotlib`; the legacy demo imports `natsort`. Installing these packages does not supply the missing project-specific modules described below. No pinned environment or original training environment is included. Older PyTorch builds may require NumPy 1.x to avoid NumPy 2 binary-compatibility warnings.

### Minimal model forward pass

Run this Python example from the repository root. It checks the model's tensor interface with random weights; it does **not** perform trained text removal.

```python
import torch
from MBE.HiCarNet import HicarNet

model = HicarNet().eval()
image = torch.rand(1, 3, 64, 64)  # RGB, batch x channels x height x width

with torch.no_grad():
    outputs = model(image)

restored, stage2, stage1, mask_prediction = outputs
print([tuple(output.shape) for output in outputs])
# [(1, 3, 64, 64), (1, 3, 64, 64), (1, 3, 64, 64), (1, 3, 64, 64)]
```

The model splits images into half- and quarter-image patches and downsamples features. Use spatial dimensions divisible by 8; padding and unpadding are the caller's responsibility. `forward(image, flag=False)` is the default interface; `flag=True` enables the conditional patch-attention paths. The fourth return value is a raw three-channel mask prediction, not a thresholded binary mask.

The example's default forward path was checked on CPU with a `1 × 3 × 64 × 64` tensor using Python 3.10, PyTorch 2.0.1, and torchvision 0.15.2. All four outputs had the displayed shape and finite values. This check does not validate training, GPU execution, image-file preprocessing, or benchmark scores.

## Data preparation

The training and validation loaders expect the following layout under each split directory:

```text
dataset/
├── train/
│   ├── all_images/    # Input images containing text
│   ├── all_labels/    # Corresponding text-free target images
│   └── mask/          # Corresponding text-region masks
└── val/
    ├── all_images/
    ├── all_labels/
    └── mask/
```

The loader sorts the three directories independently and pairs files by position. Keep the file counts, filename ordering, and spatial dimensions aligned. The loss code treats mask values of **1 as background/valid regions** and **0 as text regions to reconstruct**; masks are loaded as RGB tensors. Training uses random crops and joint geometric augmentation, while validation optionally uses a center crop controlled by `img_options['patch_size']`.

The loaders return:

- Training: `(target, input, filename, mask, input_path)`.
- Validation: `(target, input, filename, mask)`.
- Test: `(input, filename, binary_mask)`, with input images read directly from a flat directory.

Datasets are not distributed in this repository. Obtain them from their respective providers and follow their terms of use.

### Requirements for a full training or inference pipeline

Before using the data and loss modules in a complete pipeline, the following pieces must be supplied or integrated:

- `dataset_RGB.py` imports `non_max_suppression_fast` from `MPRNet`, but `MPRNet.py` is absent.
- `dataset_RGB.py` and `detect.py` import `lib.text_detection`, `lib.utils`, and `lib.config`, which are absent.
- `losses.py` loads `training.yml` when imported; that configuration file is absent. `config.py` contains defaults, not the paper's complete training recipe.
- Several utility imports are relative to the `MBE/` working directory. Use that directory or configure the Python import path when integrating those modules.
- A training loop, the feature extractor passed to `Otherloss`, and matching trained checkpoints are not provided. Checkpoint helpers expect a dictionary containing `state_dict`; resume helpers also use `epoch` and `optimizer`.

## Evaluation

[`MBE/evaluate_PSNR_SSIM.m`](MBE/evaluate_PSNR_SSIM.m) computes mean PSNR and SSIM, converting RGB images to the Y channel of YCbCr before comparison. It includes the SSIM implementation and uses MATLAB image-processing functions.

Before running it:

1. Generate restored images using a trained model and a compatible inference pipeline.
2. Set `file_path` to the result directory and `gt_path` to the matching ground-truth directory. Replace the original machine-specific `hrnet` paths and update `datasets` as needed.
3. Ensure the result and ground-truth file lists have matching counts and ordering. The script pairs images by index, not by filename lookup.

Run `evaluate_PSNR_SSIM` in MATLAB from `MBE/` after configuring those paths. The script's default result directory is `./results/SCUT-EnsText/`. This README does not report newly reproduced benchmark results.

## Citation

```bibtex
@inproceedings{Hou_2022_ACCV,
  title     = {Multi-Branch Network with Ensemble Learning for Text Removal in the Wild},
  author    = {Hou, Yujie and Chen, Jiwei and Wang, Zengfu},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2022},
  month     = dec,
  pages     = {1333--1349},
  url       = {https://openaccess.thecvf.com/content/ACCV2022/html/Hou_Multi-Branch_Network_with_Ensemble_Learning_for_Text_Removal_in_the_ACCV_2022_paper.html}
}
```

## Acknowledgments and license

The source acknowledges **Multi-Stage Progressive Image Restoration (MPRNet)** by Zamir et al. ([paper](https://arxiv.org/abs/2102.02808)). The MATLAB evaluation file retains the original SSIM attribution to Zhou Wang and collaborators.

The repository includes an [MIT license](LICENSE). Third-party code may carry additional notices or terms in its source files, including the embedded SSIM implementation; retain those notices when reusing the code.
