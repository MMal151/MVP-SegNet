import nibabel as nib
import numpy as np
import os
import pandas as pd

from src.segmentation.process.Train import configure_gpus
from src.utils.ProcessUtils import get_metrics_inference, load_model, normalize_img, perform_cca, save_img, \
    thresholding
from src.utils.CommonUtils import is_valid_file, str_to_list, str_to_tuple

CLASS_NAME = "[Process/Inference]"


def merge_patches_max(predicts, in_shape, threshold=0.5):
    prediction_lbl = np.zeros(in_shape)

    # For overlapping coordinates, the maximum probability will be used.
    for p in predicts:
        curr_patch = p[0]
        (i, j, k) = p[1]
        patch_shape = curr_patch.shape
        prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] = \
            np.maximum(prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]], curr_patch)

    return thresholding(prediction_lbl, threshold)


def merge_patches_voting(predicts, in_shape, threshold):
    prediction_lbl = np.zeros(in_shape)

    for p in predicts:
        curr_patch = p[0]
        (i, j, k) = p[1]
        patch_shape = curr_patch.shape

        # 1. Mapping values greater than threshold to +1 and values less than -1 to threshold
        # If curr_patch[i][j][k] >= threshold then the same value will be returned else -1 will be returned.
        pred = np.where(curr_patch >= threshold, curr_patch, -1)
        # If curr_patch[i][j][k] < threshold then the same value will be returned else 1 will be returned.
        pred = np.where(pred < threshold, pred, 1)

        # 2. Add all patches.
        prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] = \
            prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] + pred

    # 3. If sum > 0, then more patches "voted" for the voxel to be "1" else it will be mapped to background.
    prediction_lbl = np.where(prediction_lbl > 0, prediction_lbl, 0)
    prediction_lbl = np.where(prediction_lbl <= 0, prediction_lbl, 1)

    return prediction_lbl


def merge_patches(predicts, in_shape, strategy="max", threshold=0.5):
    if strategy == "vote":
        return merge_patches_voting(predicts, in_shape, threshold)
    else:
        return merge_patches_max(predicts, in_shape, threshold)


def inference(cfg):
    lgr = CLASS_NAME + "[inference()]"

    if cfg["alw_parallel_processing"]:
        configure_gpus(cfg["gpus"])

    model = load_model(cfg, cfg["model_path"])

    # Saving the state of some parameters that should match the training configuration.
    normalize = cfg["normalize_img"]
    patch_shape = str_to_tuple(cfg["input_shape"])
    perf_metircs = cfg["perf_metrics"]

    assert is_valid_file(cfg["data_path"]), f"{lgr}[Error]: Invalid path for CSV File."
    df_test = pd.read_csv(cfg["data_path"])

    for idx, row in df_test.iterrows():
        img_hdr = nib.load(row['X'])
        img = img_hdr.get_fdata()

        if normalize:
            img = normalize_img(img, cfg["normalize"]["technique"])

        predicts = []

        for p in str_to_list(row["patches"]):
            (i, j, k) = p  # p -> patch coordinates. Expected format: (0, 0, 0)
            # Generating image patch and reshaping it to match the size expected by the model i.e. (batch_size, H, W, D, channel)
            patch = img[i: i + patch_shape[0], j: j + patch_shape[1], k: k + patch_shape[2]].reshape(
                (1, *patch_shape, 1))
            pred = model.predict(patch, batch_size=1)
            predicts.append((pred.reshape(patch_shape), (i, j, k)))

        predicted_lbl = merge_patches(predicts, img.shape, cfg["strategy"], cfg["threshold"])
        predicted_lbl_cca = None

        if row['Y'] is not None or row['Y'] != "":
            org_lbl = nib.load(row['Y']).get_fdata()

            results = get_metrics_inference(org_lbl, predicted_lbl, perf_metircs)
            print(f"{lgr}: Performance Metrics for {row['X']}: {results}")

            if cfg["apply_cca"]:
                predicted_lbl_cca = perform_cca(predicted_lbl)

                results = get_metrics_inference(org_lbl, predicted_lbl_cca, perf_metircs)
                print(f"{lgr}: Performance Metrics for {row['X']} with CCA: {results}")

        if cfg["save_inference"]:
            save_img(predicted_lbl,
                     os.path.join("Test_Results", row['X'].split('/')[-1].split('.nii.gz')[0] + "_Pred.nii.gz"),
                     img_hdr)

            if predicted_lbl_cca is not None:
                save_img(predicted_lbl,
                         os.path.join("Test_Results", row['X'].split('/')[-1].split('.nii.gz')[0] + "_Pred_CCA.nii.gz"),
                         img_hdr)
