"""
Fine-tuning Cellpose-SAM model using prepared cellpose_train / cellpose_val folders.

Requires installed Cellpose package (pip install cellpose). This file imports
cellpose.io, cellpose.models, and cellpose.train from that installation.

For fine-tuning described in the thesis (masked loss, possibility to change learning rate schedule), replace
the installed train.py in Cellpose library with the provided train.py file from thesis archive before running
this script; see the user guide.

Example::

    python run_finetune_cellpose_sam.py \\
        --data-root path/to/prepared_out \\
        --output-dir path/to/exp_run_001 \\
        --gpu --gpu-device 1

Seed: sets Python random, NumPy, and PyTorch seeds before building the model and loading
data for more reproducible runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing cellpose_train/ and cellpose_val/ (from prepare script)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write logs, losses.npz, run_config.json, checkpoints/",
    )
    ap.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help=(
            "Path to weights file. When omitted, it uses default 'cpsam' model. "
        ),
    )
    ap.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    ap.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device index (e.g. 0, 1). Ignored without --gpu (CPU is used).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument(
        "--lr-decay-start-epoch",
        type=int,
        default=None,
        help=(
            "Optional epoch index at which the end-of-training LR halving schedule begins. "
            "Example: with --epochs 100 and --lr-decay-start-epoch 80, LR decay starts at epoch 80 "
            "instead of the legacy start at epoch 50."
        ),
    )
    ap.add_argument("--min-train-masks", type=int, default=1)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for random, NumPy, and PyTorch before training (reproducibility). "
            "train_seg still reseeds NumPy per epoch internally."
        ),
    )
    ap.add_argument("--no-masked-loss", action="store_true")
    ap.add_argument("--no-thesis-checkpoints", action="store_true")
    ap.add_argument(
        "--look-one-level-down",
        action="store_true",
        help=(
            "Also load images from immediate subfolders of cellpose_train/ and cellpose_val/ "
            "(Cellpose get_image_files). Default is off: only files directly in those folders "
            "(matches flat thesis attachment layouts)."
        ),
    )
    args = ap.parse_args()

    try:
        from cellpose import io, models, train
    except ImportError as e:
        print(
            "ERROR: cellpose is not installed. Install with: pip install cellpose\n"
            f"Detail: {e}",
            file=sys.stderr,
        )
        return 1

    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    data_root = args.data_root.resolve()
    train_dir = data_root / "cellpose_train"
    val_dir = data_root / "cellpose_val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        print(f"ERROR: expected {train_dir} and {val_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device, _gpu = models.assign_device(
        use_torch=True, gpu=args.gpu, device=args.gpu_device
    )

    pretrained = args.pretrained_model if args.pretrained_model is not None else "cpsam"

    model = models.CellposeModel(device=device, pretrained_model=pretrained)

    images, labels, image_names, test_images, test_labels, image_names_test = (
        io.load_train_test_data(
            str(train_dir),
            str(val_dir),
            image_filter=None,
            mask_filter="_masks",
            look_one_level_down=args.look_one_level_down,
        )
    )

    aug_notes = (
        "random_rotate_and_resize (2D): flip p=0.5, rotation uniform [0,2pi], "
        "scale_range default 0.5 on [0.75,1.25] before crop; output bsize=256; "
        "Transformer rdrop=0.4 during train. See cellpose.transforms and cellpose.vit_sam."
    )

    t0 = time.time()
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        images,
        labels,
        train_files=image_names,
        test_data=test_images,
        test_labels=test_labels,
        test_files=image_names_test,
        load_files=True,
        normalize=True,
        rescale=False,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        min_train_masks=args.min_train_masks,
        save_path=str(output_dir),
        save_every=max(1, args.epochs // 2),
        save_each=False,
        bsize=256,
        scale_range=0.5,
        model_name="cellposeSAM_finetune_full_epochs",
        masked_loss=not args.no_masked_loss,
        validate_every_epoch=True,
        experiment_dir=str(output_dir),
        thesis_checkpoints=not args.no_thesis_checkpoints,
        plateau_patience=5,
        val_improve_eps=1e-6,
        augmentation_notes=aug_notes,
        lr_decay_start_epoch=args.lr_decay_start_epoch,
    )

    extra = {
        "pretrained_model": pretrained,
        "gpu": args.gpu,
        "gpu_device": args.gpu_device,
        "device": str(device),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "look_one_level_down": args.look_one_level_down,
        "wall_clock_sec": time.time() - t0,
        "final_model_path": str(model_path),
        "random_seed": args.seed,
    }
    with open(output_dir / "run_driver_meta.json", "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2)

    print(f"Done. Model: {model_path}")
    print(f"Logs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
