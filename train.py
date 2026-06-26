import time
import os
import json
import numpy as np
from cellpose import io, utils, models, dynamics
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

import logging

train_logger = logging.getLogger(__name__)

def _log_event(msg: str, log_path: Path | None = None, level: str = "info") -> None:
    """Log to the configured logger and optionally append to training.log."""
    if level == "warning":
        train_logger.warning(msg)
    elif level == "critical":
        train_logger.critical(msg)
    else:
        train_logger.info(msg)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

def _loss_fn_class(lbl, y, class_weights=None):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (numpy.ndarray): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        
    Returns:
        torch.Tensor: Loss value.

    """

    criterion3 = nn.CrossEntropyLoss(reduction="mean", weight=class_weights)
    loss3 = criterion3(y[:, :-3], lbl[:, 0].long())
    
    return loss3

def _loss_fn_seg(lbl, y, device, masked_loss=False):
    """
    Calculates the loss function between true labels lbl and prediction y.

    Args:
        lbl (torch.Tensor): True labels (cellprob, flowsY, flowsX).
        y (torch.Tensor): Predicted values (flowsY, flowsX, cellprob).
        device (torch.device): Device on which the tensors are located.
        masked_loss: If True, restrict flow MSE and cellprob BCE to pixels where
            lbl[:, -3] > 0.5 (annotated foreground). BCE uses only those pixels
            (no background negatives).

    Returns:
        torch.Tensor: Loss value.

    """
    if not masked_loss:
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5. * lbl[:, -2:]
        loss = criterion(y[:, -3:-1], veci)
        loss /= 2.
        loss2 = criterion2(y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype))
        loss = loss + loss2
        return loss

    valid = (lbl[:, -3] > 0.5).to(y.dtype)
    nv = valid.sum()
    if nv < 1e-6:
        return (y[:, 0] * 0).sum()

    veci = 5. * lbl[:, -2:]
    se = (y[:, -3:-1] - veci) ** 2
    mse_map = se.mean(dim=1)
    flow_loss = 0.5 * (mse_map * valid).sum() / (nv + 1e-8)

    tgt = (lbl[:, -3] > 0.5).to(y.dtype)
    bce_map = F.binary_cross_entropy_with_logits(y[:, -1], tgt, reduction="none")
    bce_loss = (bce_map * valid).sum() / (nv + 1e-8)
    return flow_loss + bce_loss

def _reshape_norm(data, channel_axis=None, normalize_params={"normalize": False}):
    """
    Reshapes and normalizes the input data.

    Args:
        data (list): List of input data, with channels axis first or last.
        normalize_params (dict, optional): Dictionary of normalization parameters. Defaults to {"normalize": False}.

    Returns:
        list: List of reshaped and normalized data.
    """
    if (np.array([td.ndim!=3 for td in data]).sum() > 0 or
        np.array([td.shape[0]!=3 for td in data]).sum() > 0):
        data_new = []
        for td in data:
            if td.ndim == 3:
                channel_axis0 = channel_axis if channel_axis is not None else np.array(td.shape).argmin()
                # put channel axis first 
                td = np.moveaxis(td, channel_axis0, 0)
                td = td[:3] # keep at most 3 channels
            if td.ndim == 2 or (td.ndim == 3 and td.shape[0] == 1):
                td = np.stack((td, 0*td, 0*td), axis=0)
            elif td.ndim == 3 and td.shape[0] < 3:
                td = np.concatenate((td, 0*td[:1]), axis=0)
            data_new.append(td)
        data = data_new
    if normalize_params["normalize"]:
        data = [
            normalize_img(td, normalize=normalize_params, axis=0)
            for td in data
        ]
    return data

def _get_batch(inds, data=None, labels=None, files=None, labels_files=None,
               normalize_params={"normalize": False}):
    """
    Get a batch of images and labels.

    Args:
        inds (list): List of indices indicating which images and labels to retrieve.
        data (list or None): List of image data. If None, images will be loaded from files.
        labels (list or None): List of label data. If None, labels will be loaded from files.
        files (list or None): List of file paths for images.
        labels_files (list or None): List of file paths for labels.
        normalize_params (dict): Dictionary of parameters for image normalization (will be faster, if loading from files to pre-normalize).

    Returns:
        tuple: A tuple containing two lists: the batch of images and the batch of labels.
    """
    if data is None:
        lbls = None
        imgs = [io.imread(files[i]) for i in inds]
        imgs = _reshape_norm(imgs, normalize_params=normalize_params)
        if labels_files is not None:
            lbls = [io.imread(labels_files[i])[1:] for i in inds]
    else:
        imgs = [data[i] for i in inds]
        lbls = [labels[i][1:] for i in inds]
    return imgs, lbls

def _reshape_norm_save(files, channels=None, channel_axis=None,
                       normalize_params={"normalize": False}):
    """ not currently used -- normalization happening on each batch if not load_files """
    files_new = []
    for f in trange(files):
        td = io.imread(f)
        if channels is not None:
            td = convert_image(td, channels=channels,
                                          channel_axis=channel_axis)
            td = td.transpose(2, 0, 1)
        if normalize_params["normalize"]:
            td = normalize_img(td, normalize=normalize_params, axis=0)
        fnew = os.path.splitext(str(f))[0] + "_cpnorm.tif"
        io.imsave(fnew, td)
        files_new.append(fnew)
    return files_new
    # else:
    #     train_files = reshape_norm_save(train_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)
    # elif test_files is not None:
    #     test_files = reshape_norm_save(test_files, channels=channels,
    #                     channel_axis=channel_axis, normalize_params=normalize_params)


def _process_train_test(train_data=None, train_labels=None, train_files=None,
                        train_labels_files=None, train_probs=None, test_data=None,
                        test_labels=None, test_files=None, test_labels_files=None,
                        test_probs=None, load_files=True, min_train_masks=5,
                        compute_flows=False, normalize_params={"normalize": False}, 
                        channel_axis=None, device=None):
    """
    Process train and test data.

    Args:
        train_data (list or None): List of training data arrays.
        train_labels (list or None): List of training label arrays.
        train_files (list or None): List of training file paths.
        train_labels_files (list or None): List of training label file paths.
        train_probs (ndarray or None): Array of training probabilities.
        test_data (list or None): List of test data arrays.
        test_labels (list or None): List of test label arrays.
        test_files (list or None): List of test file paths.
        test_labels_files (list or None): List of test label file paths.
        test_probs (ndarray or None): Array of test probabilities.
        load_files (bool): Whether to load data from files.
        min_train_masks (int): Minimum number of masks required for training images.
        compute_flows (bool): Whether to compute flows.
        channels (list or None): List of channel indices to use.
        channel_axis (int or None): Axis of channel dimension.
        rgb (bool): Convert training/testing images to RGB.
        normalize_params (dict): Dictionary of normalization parameters.
        device (torch.device): Device to use for computation.

    Returns:
        tuple: A tuple containing the processed train and test data and sampling probabilities and diameters.
    """
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else None
    
    if train_data is not None and train_labels is not None:
        # if data is loaded
        nimg = len(train_data)
        nimg_test = len(test_data) if test_data is not None else None
    else:
        # otherwise use files
        nimg = len(train_files)
        if train_labels_files is None:
            train_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in train_files
            ]
            train_labels_files = [tf for tf in train_labels_files if os.path.exists(tf)]
        if (test_data is not None or
                test_files is not None) and test_labels_files is None:
            test_labels_files = [
                os.path.splitext(str(tf))[0] + "_flows.tif" for tf in test_files
            ]
            test_labels_files = [tf for tf in test_labels_files if os.path.exists(tf)]
        if not load_files:
            train_logger.info(">>> using files instead of loading dataset")
        else:
            # load all images
            train_logger.info(">>> loading images and labels")
            train_data = [io.imread(train_files[i]) for i in trange(nimg)]
            train_labels = [io.imread(train_labels_files[i]) for i in trange(nimg)]
        nimg_test = len(test_files) if test_files is not None else None
        if load_files and nimg_test:
            test_data = [io.imread(test_files[i]) for i in trange(nimg_test)]
            test_labels = [io.imread(test_labels_files[i]) for i in trange(nimg_test)]

    ### check that arrays are correct size
    if ((train_labels is not None and nimg != len(train_labels)) or
        (train_labels_files is not None and nimg != len(train_labels_files))):
        error_message = "train data and labels not same length"
        train_logger.critical(error_message)
        raise ValueError(error_message)
    if ((test_labels is not None and nimg_test != len(test_labels)) or
        (test_labels_files is not None and nimg_test != len(test_labels_files))):
        train_logger.warning("test data and labels not same length, not using")
        test_data, test_files = None, None
    if train_labels is not None:
        if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
            error_message = "training data or labels are not at least two-dimensional"
            train_logger.critical(error_message)
            raise ValueError(error_message)
        if train_data[0].ndim > 3:
            error_message = "training data is more than three-dimensional (should be 2D or 3D array)"
            train_logger.critical(error_message)
            raise ValueError(error_message)

    ### check that flows are computed
    if train_labels is not None:
        train_labels = dynamics.labels_to_flows(train_labels, files=train_files,
                                                device=device)
        if test_labels is not None:
            test_labels = dynamics.labels_to_flows(test_labels, files=test_files,
                                                   device=device)
    elif compute_flows:
        for k in trange(nimg):
            dynamics.labels_to_flows(
                [io.imread(train_labels_files[k])],
                files=[train_files[k]] if train_files is not None else None,
                device=device,
            )
        if test_files is not None:
            for k in trange(nimg_test):
                dynamics.labels_to_flows(
                    [io.imread(test_labels_files[k])],
                    files=[test_files[k]] if test_files is not None else None,
                    device=device,
                )

    ### compute diameters
    nmasks = np.zeros(nimg)
    diam_train = np.zeros(nimg)
    train_logger.info(">>> computing diameters")
    for k in trange(nimg):
        tl = (train_labels[k][0]
              if train_labels is not None else io.imread(train_labels_files[k])[0])
        diam_train[k], dall = utils.diameters(tl)
        nmasks[k] = len(dall)
    diam_train[diam_train < 5] = 5.
    if test_data is not None:
        diam_test = np.array(
            [utils.diameters(test_labels[k][0])[0] for k in trange(len(test_labels))])
        diam_test[diam_test < 5] = 5.
    elif test_labels_files is not None:
        diam_test = np.array([
            utils.diameters(io.imread(test_labels_files[k])[0])[0]
            for k in trange(len(test_labels_files))
        ])
        diam_test[diam_test < 5] = 5.
    else:
        diam_test = None

    ### check to remove training images with too few masks
    if min_train_masks > 0:
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            _log_event(
                f"{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set",
                level="warning",
            )
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            if train_data is not None:
                train_data = [train_data[i] for i in ikeep]
                train_labels = [train_labels[i] for i in ikeep]
            if train_files is not None:
                train_files = [train_files[i] for i in ikeep]
            if train_labels_files is not None:
                train_labels_files = [train_labels_files[i] for i in ikeep]
            if train_probs is not None:
                train_probs = train_probs[ikeep]
            diam_train = diam_train[ikeep]
            nimg = len(train_data)

    ### normalize probabilities
    train_probs = 1. / nimg * np.ones(nimg,
                                      "float64") if train_probs is None else train_probs
    train_probs /= train_probs.sum()
    if test_files is not None or test_data is not None:
        test_probs = 1. / nimg_test * np.ones(
            nimg_test, "float64") if test_probs is None else test_probs
        test_probs /= test_probs.sum()

    ### reshape and normalize train / test data
    normed = False
    if normalize_params["normalize"]:
        _log_event(f">>> normalizing {normalize_params}")
    if train_data is not None:
        train_data = _reshape_norm(train_data, channel_axis=channel_axis, 
                                   normalize_params=normalize_params)
        normed = True
    if test_data is not None:
        test_data = _reshape_norm(test_data, channel_axis=channel_axis,
                                  normalize_params=normalize_params)

    return (train_data, train_labels, train_files, train_labels_files, train_probs,
            diam_train, test_data, test_labels, test_files, test_labels_files,
            test_probs, diam_test, normed)


def _run_validation_loss(
    net,
    rperm,
    test_data,
    test_labels,
    test_files,
    test_labels_files,
    kwargs,
    batch_size,
    diam_test,
    rescale,
    net_diam_mean,
    scale_range,
    bsize,
    class_weights,
    masked_loss,
):
    """Mean loss over the validation permutation (same random_rotate_and_resize as train)."""
    device = net.device
    nimg_test = len(rperm)
    lavgt = 0.0
    net.eval()
    with torch.no_grad():
        for ibatch in range(0, len(rperm), batch_size):
            inds = rperm[ibatch : ibatch + batch_size]
            imgs, lbls = _get_batch(
                inds,
                data=test_data,
                labels=test_labels,
                files=test_files,
                labels_files=test_labels_files,
                **kwargs,
            )
            diams = np.array([diam_test[i] for i in inds])
            rsc = diams / net_diam_mean if rescale else np.ones(len(diams), "float32")
            imgi, lbl = random_rotate_and_resize(
                imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize)
            )[:2]
            X = torch.from_numpy(imgi).to(device)
            lbl_t = torch.from_numpy(lbl).to(device)

            if X.dtype != net.dtype:
                X = X.to(net.dtype)
                lbl_t = lbl_t.to(net.dtype)

            y = net(X)[0]
            loss = _loss_fn_seg(lbl_t, y, device, masked_loss=masked_loss)
            if y.shape[1] > 3:
                loss = loss + _loss_fn_class(lbl_t, y, class_weights=class_weights)
            test_loss = loss.item() * len(imgi)
            lavgt += test_loss
    lavgt /= max(1, nimg_test)
    return lavgt


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_seg(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=5e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None,
              masked_loss=False, validate_every_epoch=False,
              experiment_dir=None, thesis_checkpoints=False,
              plateau_patience=5, val_improve_eps=1e-6,
              augmentation_notes="",
              lr_decay_start_epoch=None, lr_decay_n_halvings=5, lr_decay_step_epochs=None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train. If `net` is a bfloat16 model on MPS, it will be converted to float32 for training. The saved models will be in float32, but the original model will be returned in bfloat16 for consistency. CUDA/CPU will train in bfloat16 if that is the provided net dtype.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Deprecated in v4.0.1+ - AdamW always used.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        save_each (bool, optional): Boolean - save the network to a new filename at every [save_each] epoch. Defaults to False.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): If True, diameter-based rescale in augmentations
            (``diam_i / net.diam_mean``). Defaults to False (matches ``diameter=None`` inference).
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.
        masked_loss (bool, optional): Use foreground-only masked loss (see ``_loss_fn_seg``).
        validate_every_epoch (bool, optional): Run validation every epoch (not only 5,10,...).
        experiment_dir (str or Path, optional): If set, write ``run_config.json``, ``training.log``,
            and ``losses.npz`` (updated each epoch) for experiment tracking.
        thesis_checkpoints (bool, optional): If True with ``experiment_dir``, save ``best_model.pt``,
            midpoint checkpoint, and plateau checkpoints (requires validation set).
        plateau_patience (int): Consecutive epochs without val improvement before plateau save.
        val_improve_eps (float): Minimum relative improvement to reset plateau counter.
        augmentation_notes (str): Free-text description stored in ``run_config.json``.
        lr_decay_start_epoch (int or None): If set, start the end-of-training step-halving schedule
            at this epoch index (0-based). This overrides the legacy behavior that starts decay
            at epoch 50 when n_epochs=100 (i.e. n_epochs-50).
        lr_decay_n_halvings (int): Number of halving steps in the decay tail (default 10).
        lr_decay_step_epochs (int or None): Epochs per halving step. If None, it is chosen so the
            decay tail spans the remaining epochs after lr_decay_start_epoch.

    Returns:
        tuple: A tuple containing the path to the saved model weights, training losses, and test losses.
       
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    original_net_dtype = None
    if device.type == 'mps' and net.dtype == torch.bfloat16:
        # NOTE: this produces a side effect of returning a network that is not of a guaranteed dtype \
        original_net_dtype = torch.bfloat16 
        train_logger.warning("Training with bfloat16 on MPS is not supported, using float32 network instead")
        net.dtype = torch.float32
        net.to(torch.float32)

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = _process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule (explicit array indexed by epoch)
    # Base: 10-epoch linear warmup then constant LR.
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))

    # Optional override: choose when the end-of-training halving schedule begins.
    # Legacy behavior (kept when lr_decay_start_epoch is None):
    #   - if n_epochs > 99: last 50 epochs replaced by 10 halvings of 5 epochs each
    #   - if n_epochs > 300: last 100 epochs replaced by 10 halvings of 10 epochs each
    if lr_decay_start_epoch is not None:
        s = int(lr_decay_start_epoch)
        if s < 0 or s >= n_epochs:
            raise ValueError(
                f"lr_decay_start_epoch must be in [0, n_epochs-1], got {s} for n_epochs={n_epochs}"
            )
        tail = n_epochs - s
        if tail > 0 and int(lr_decay_n_halvings) > 0:
            step_epochs = (
                int(lr_decay_step_epochs)
                if lr_decay_step_epochs is not None
                else max(1, tail // int(lr_decay_n_halvings))
            )
            if step_epochs <= 0:
                raise ValueError(f"lr_decay_step_epochs must be >= 1, got {step_epochs}")
            LR = LR[:s]
            lr_cur = float(LR[-1]) if len(LR) else float(learning_rate)
            remain = tail
            for _ in range(int(lr_decay_n_halvings)):
                if remain <= 0:
                    break
                lr_cur = lr_cur / 2.0
                nstep = min(step_epochs, remain)
                LR = np.append(LR, lr_cur * np.ones(nstep))
                remain -= nstep
            if remain > 0:
                LR = np.append(LR, lr_cur * np.ones(remain))
    else:
        if n_epochs > 300:
            LR = LR[:-100]
            for _ in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        elif n_epochs > 99:
            LR = LR[:-50]
            for _ in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    _log_event(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    _log_event(f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}")
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    t0 = time.time()
    # Default to a Cellpose-SAM-specific prefix to avoid confusion with classic Cellpose runs.
    # Include timestamp to prevent accidental overwrites across runs.
    model_name = f"cellposeSAM_full_epochs_{int(t0)}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    exp_dir = Path(experiment_dir).resolve() if experiment_dir is not None else None
    ckpt_dir = None
    log_path = None
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "training.log"
        if thesis_checkpoints:
            ckpt_dir = exp_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    # echo key run header into training.log if enabled
    _log_event(f">>> saving model to {filename}", log_path=log_path)

    lr_idx_max = len(LR) - 1
    run_config = _json_safe(
        {
            "loss": "MSE(flows)*0.5 + BCE(cellprob) per _loss_fn_seg; "
            + ("masked foreground-only" if masked_loss else "dense (full image)"),
            "masked_loss": masked_loss,
            "masked_loss_description": (
                "Flow MSE and cellprob BCE averaged only over pixels with "
                "ground-truth foreground (lbl[:, -3] > 0.5). BCE has no background term."
                if masked_loss
                else None
            ),
            "normalize": normalize if isinstance(normalize, (bool, str)) else _json_safe(normalize),
            "rescale": rescale,
            "scale_range": scale_range,
            "bsize": bsize,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "nimg_train": nimg,
            "nimg_test": nimg_test,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "min_train_masks": min_train_masks,
            "validate_every_epoch": validate_every_epoch,
            "thesis_checkpoints": thesis_checkpoints,
            "plateau_patience": plateau_patience,
            "val_improve_eps": val_improve_eps,
            "save_every": save_every,
            "save_each": save_each,
            "model_name": str(model_name),
            "save_path": str(save_path),
            "experiment_dir": str(exp_dir) if exp_dir else None,
            "augmentation_notes": augmentation_notes
            or (
                "random_rotate_and_resize: random rotation uniform [0,2pi], "
                "horizontal flip p=0.5, scale uniform in [1-scale_range/2, 1+scale_range/2] "
                f"(scale_range={scale_range}), random crop to (bsize,bsize); "
                "Transformer uses stochastic depth (rdrop) during training."
            ),
            "lr_schedule": "10-epoch linear warmup 0->lr; tail per n_epochs>99 or >300 (or overridden by lr_decay_start_epoch)",
            "lr_decay_start_epoch": lr_decay_start_epoch,
            "lr_decay_n_halvings": lr_decay_n_halvings,
            "lr_decay_step_epochs": lr_decay_step_epochs,
        }
    )
    if exp_dir is not None:
        with open(exp_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    train_losses = np.zeros(n_epochs, dtype=np.float64)
    test_losses = np.full(n_epochs, np.nan, dtype=np.float64)
    best_val = float("inf")
    epochs_since_improve = 0
    mid_saved = False
    has_val = test_data is not None or test_files is not None

    for iepoch in range(n_epochs):
        epoch_t0 = time.time()
        lavg, nsum = 0, 0
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(
                np.arange(0, nimg), size=(nimg_per_epoch,), p=train_probs
            )
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        lr_now = LR[min(iepoch, lr_idx_max)]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now
        net.train()
        for k in range(0, nimg_per_epoch, batch_size):
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = _get_batch(
                inds,
                data=train_data,
                labels=train_labels,
                files=train_files,
                labels_files=train_labels_files,
                **kwargs,
            )
            diams = np.array([diam_train[i] for i in inds])
            rsc = (
                diams / net.diam_mean.item()
                if rescale
                else np.ones(len(diams), "float32")
            )
            imgi, lbl = random_rotate_and_resize(
                imgs, Y=lbls, rescale=rsc, scale_range=scale_range, xy=(bsize, bsize)
            )[:2]
            X = torch.from_numpy(imgi).to(device)
            lbl_t = torch.from_numpy(lbl).to(device)

            if X.dtype != net.dtype:
                X = X.to(net.dtype)
                lbl_t = lbl_t.to(net.dtype)

            y = net(X)[0]
            loss = _loss_fn_seg(lbl_t, y, device, masked_loss=masked_loss)
            if y.shape[1] > 3:
                loss = loss + _loss_fn_class(lbl_t, y, class_weights=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item() * len(imgi)
            lavg += train_loss
            nsum += len(imgi)
            train_losses[iepoch] += train_loss
        train_losses[iepoch] /= max(1, nimg_per_epoch)
        train_epoch_mean = train_losses[iepoch]

        run_val = validate_every_epoch or (iepoch == 5 or iepoch % 10 == 0)
        lavgt = float("nan")
        if run_val and has_val and nimg_test:
            np.random.seed(42)
            if nimg_test != nimg_test_per_epoch:
                rperm_val = np.random.choice(
                    np.arange(0, nimg_test),
                    size=(nimg_test_per_epoch,),
                    p=test_probs,
                )
            else:
                rperm_val = np.random.permutation(np.arange(0, nimg_test))
            lavgt = _run_validation_loss(
                net,
                rperm_val,
                test_data,
                test_labels,
                test_files,
                test_labels_files,
                kwargs,
                batch_size,
                diam_test,
                rescale,
                net.diam_mean.item(),
                scale_range,
                bsize,
                class_weights,
                masked_loss,
            )
            test_losses[iepoch] = lavgt

        epoch_sec = time.time() - epoch_t0
        gpu_m = ""
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
                gpu_m = f", gpu_mem_peak_mb={torch.cuda.max_memory_allocated(device) / 1e6:.1f}"
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                gpu_m = ""

        log_line = (
            f"epoch={iepoch} train_loss={train_epoch_mean:.6f} val_loss={lavgt:.6f} "
            f"lr={lr_now:.8f} epoch_sec={epoch_sec:.2f} wall_sec={time.time()-t0:.2f}{gpu_m}"
        )
        _log_event(log_line, log_path=log_path)
        if log_path is not None:
            np.savez(
                exp_dir / "losses.npz",
                train_losses=train_losses,
                test_losses=test_losses,
            )

        if ckpt_dir is not None and has_val and not np.isnan(lavgt):
            if lavgt < best_val - val_improve_eps:
                best_val = lavgt
                epochs_since_improve = 0
                best_path = ckpt_dir / "best_model.pt"
                _log_event(
                    f"saving best model (epoch={iepoch}) val_loss={lavgt:.6f} -> {best_path}",
                    log_path=log_path,
                )
                _log_event(f"updating best pointer -> {best_path}", log_path=log_path)
                net.save_model(best_path)
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= plateau_patience:
                    ppath = ckpt_dir / f"plateau_epoch_{iepoch:04d}_no_improve_{plateau_patience}.pt"
                    _log_event(
                        f"validation plateau {plateau_patience} epochs -> {ppath}",
                        log_path=log_path,
                    )
                    net.save_model(ppath)
                    epochs_since_improve = 0

        if (
            ckpt_dir is not None
            and n_epochs > 1
            and not mid_saved
            and iepoch >= max(0, n_epochs // 2)
        ):
            mid_path = ckpt_dir / f"midpoint_epoch_{iepoch:04d}.pt"
            _log_event(f"midpoint checkpoint -> {mid_path}", log_path=log_path)
            net.save_model(mid_path)
            mid_saved = True

        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)

    net.save_model(filename)

    if original_net_dtype is not None:
        net.dtype = original_net_dtype
        net.to(original_net_dtype)

    return filename, train_losses, test_losses
