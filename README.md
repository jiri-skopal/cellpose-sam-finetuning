# Instance Segmentation of Membrane-Stained Cells Using Cellpose-SAM

Master's thesis project at **Masaryk University, Faculty of Informatics**.

This repository accompanies the thesis [*Instance Segmentation of Membrane-Stained Cells Using Cellpose-SAM*](https://is.muni.cz/th/wn5oz/). It contains code to **fine-tune** [Cellpose-SAM](https://github.com/MouseLand/cellpose) on **3D microscopy volumes** of membrane-stained cells — a setting where out-of-the-box models often underperform and domain-specific adaptation helps.

## What this project does

Accurate **instance segmentation** (separating individual cells in dense 3D tissue) is a prerequisite for quantitative analysis in cell biology. This work explores fine-tuning Cellpose-SAM — a foundation model for cell segmentation — on membrane-stained 3D data, including:

- **Custom training logic** (`train.py`) with masked loss, validation-driven checkpoints, and structured logging
- **A fine-tuning driver** (`run_finetune_cellpose_sam.py`) that wires data loading, hyperparameters, and outputs into a single workflow
- **An example 3D volume** (`image_data/example.tif`) for trying inference without preparing your own data first

The **fine-tuned model weights** (`finetuned_CellposeSAM.pt`) are not stored here; they are included in the **thesis attachment** on the thesis page linked below.

## How to use this repository

Installation, inference commands, fine-tuning setup, and data format are documented in **[docs/USAGE.md](docs/USAGE.md)**.

## Citation

```bibtex
@MastersThesis{Skopal2026thesis,
  AUTHOR = "Skopal, Jiří",
  TITLE = "Instance Segmentation of Membrane-Stained Cells Using Cellpose-SAM [online]",
  YEAR = "2026 [cit. 2026-06-25]",
  TYPE = "Diplomová práce",
  SCHOOL = "Masarykova univerzita, Fakulta informatiky, Brno",
  NOTE = "SUPERVISOR : Martin Maška",
  URL = "https://is.muni.cz/th/wn5oz/",
}