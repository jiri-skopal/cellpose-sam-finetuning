# Usage Guide

This document describes how to use the code in this repository for **inference** and **fine-tuning** with Cellpose-SAM.

> **Note on commands:** change the line-continuation characters according to your OS/shell (e.g., `\` in bash, `^` in Windows Command Prompt).

## Repository contents

This repository includes:

- `run_finetune_cellpose_sam.py` — wrapper for running fine-tuning
- `train.py` — modified from the `cellpose` package; required for thesis-specific training (masked loss, checkpoints, logging)
- `requirements.txt` — Python dependencies
- `image_data/example.tif` — example 3D volume for inference
- `image_data/cellpose_train/` and `image_data/cellpose_val/` — empty folders for your own training and validation data

The fine-tuned model **`finetuned_CellposeSAM.pt`** is **not** in this GitHub repository. Download it from the thesis attachment on the [thesis page](https://is.muni.cz/th/wn5oz/) and place it in the repository root (or pass its path via `--pretrained_model`).

## Installation

The required steps for inference or fine-tuning follow the official Cellpose GitHub repository: `https://github.com/MouseLand/cellpose`.

It is recommended to have **Python 3.8+** and (as a first step) to create a Python virtual environment. Then:

1. (Optional) For GPU usage (faster inference / fine-tuning), install **PyTorch with CUDA** if possible:

```bash
pip install torch torchvision --index-url \
    https://download.pytorch.org/whl/cu124
```

For another CUDA version, follow: `https://pytorch.org/get-started/locally/`.

2. Install the other required packages from the provided `requirements.txt` by running (from the repository root):

```bash
pip install -r requirements.txt
```

For the reference installation, **Python 3.11.2** and **pip 23.0.1** were used.

## Inference

Once the environment is prepared and packages are installed, inference is ready.

Assuming you are in the repository root, a basic inference command using the **default Cellpose-SAM model** (downloaded automatically on first run) is:

```bash
python -m cellpose \
    --image_path image_data/example.tif \
    --use_gpu \
    --do_3D \
    --save_tif
```

### Inference command hyperparameters

Cellpose-SAM supports many parameters; the main ones used for inference include:

- `--image_path`: path to the input image to be segmented
- `--use_gpu`: use GPU if PyTorch with CUDA is installed
- `--gpu_device`: which GPU device to use (default `0`)
- `--verbose`: show information about running and save it to a log
- `--pretrained_model`: path to the pretrained model to use; if omitted, the default model is used
- `--do_3D`: run 3D segmentation
- `--cellprob_threshold`: sets the `t_cp` hyperparameter (default `0.0`)
- `--flow_threshold`: sets the `t_flow` hyperparameter (default `0.4`)
- `--flow3D_smooth`: sets the `sigma_flow` hyperparameter (default `0.0`)
- `--min_size`: sets the `s_min` hyperparameter (default `15`)
- `--save_tif`: save predicted masks as a `.tif`
- `--savedir`: output folder for results (defaults to the input image folder)
- `--no_npy`: suppress saving of the `.npy` file

To run inference with the **fine-tuned model** (from the thesis attachment) and tuned hyperparameters, execute:

```bash
python -m cellpose \
    --image_path image_data/example.tif \
    --use_gpu \
    --verbose \
    --pretrained_model finetuned_CellposeSAM.pt \
    --do_3D \
    --cellprob_threshold -2.0 \
    --flow3D_smooth 0.5 \
    --min_size 200 \
    --save_tif \
    --no_npy
```

The predicted masks are saved in `image_data/example_cp_masks.tif` as a 16-bit image.

## Fine-tuning

To reproduce (or run a similar) fine-tuning setup, the provided wrapper `run_finetune_cellpose_sam.py` is used.

However, the training functionality in the installed `cellpose` package does **not** include the thesis-specific configuration (e.g., masked loss, logging). Therefore, it is necessary to replace the installed `train.py` with the provided `train.py` from this repository.

If the virtual environment was used for installation, the default `train.py` to be replaced is located at:

- `name_of_the_venv/Lib/site-packages/cellpose/train.py` (Windows), or
- `name_of_the_venv/lib/pythonX.Y/site-packages/cellpose/train.py` (Linux/macOS).

### Fine-tuning hyperparameters

The wrapper `run_finetune_cellpose_sam.py` exposes learning hyperparameters such as:

- `--data-root`: folder containing `cellpose_train` and `cellpose_val` with training and validation samples
- `--output-dir`: output folder where to save logs, losses, and checkpoints
- `--pretrained-model`: path to the pretrained model; default model if omitted
- `--gpu`: use GPU if PyTorch with CUDA is installed
- `--gpu-device`: GPU device index (default `0`)
- `--epochs`: number of training epochs (default `100`)
- `--batch-size`: batch size (default `8`)
- `--lr`: learning rate (default `0.00001`)
- `--weight-decay`: weight decay (default `0.1`)
- `--lr-decay-start-epoch`: epoch at which to start the learning-rate halving tail; default Cellpose-SAM behaviour if omitted
- `--min-train-masks`: minimum masks per image to include in training
- `--no-masked-loss`: disable masked loss and use the default Cellpose-SAM loss
- `--no-thesis-checkpoints`: disable saving of (i) best-model checkpoints on validation improvement and (ii) plateau checkpoints when validation does not improve for a fixed number of epochs

### Run fine-tuning

As the reference annotations cannot be shared publicly because the sequences may extend the Cell Tracking Challenge dataset repository later, `cellpose_train` and `cellpose_val` folders are empty in this repository. Before running fine-tuning, fill these folders with your own image–mask pairs. Each image named `basename.tif` expects a corresponding mask file `basename_masks.tif`. Mask instances must be labelled by unique positive labels, with the background having a zero label. Ideally, the annotations should follow the standard CTC format of annotations (`https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf`) as the fine-tuning described in the thesis was conducted with such a format.

Fine-tuning can be run with:

```bash
python run_finetune_cellpose_sam.py \
    --data-root image_data/ \
    --output-dir finetuning_output/ \
    --gpu
```

This command produces:

- final model after all epochs in `finetuning_output/models/`
- saved checkpoints in `finetuning_output/checkpoints/`
- training and validation losses in `finetuning_output/losses.npz`
- training log in `finetuning_output/training.log`
- fine-tuning configuration in `finetuning_output/run_config.json`
- run metadata (e.g. wall time) in `finetuning_output/run_driver_meta.json`
- computed flows in `cellpose_train` and `cellpose_val` (standard behaviour of Cellpose-SAM)
