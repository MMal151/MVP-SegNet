import os
import pandas as pd
import SimpleITK as sitk

from src.misc.CarveMix import generate_new_sample
from src.utils.CommonUtils import get_random_index

CLASS_NAME = "[misc/executor]"


def targeted_augmentation(csv_file, img_ext="_T1.nii.gz", lesion_ext="_LESION.nii.gz"):
    lgr = CLASS_NAME + "[targeted_augmentation()]"

    df = pd.read_csv(csv_file)

    img_1 = df["X_1"]
    img_2 = df["X_2"]
    lbl_1 = df["Y_1"]
    lbl_2 = df["Y_2"]

    assert len(img_1) == len(img_2) == len(lbl_1) == len(lbl_2), f"{lgr}: Number of rows are not equal."

    for i in range(len(img_1)):
        img, lbl, _, _ = generate_new_sample(img_1[i], img_2[i], lbl_1[i], lbl_2[i])

        out_path = img_1[i].split("/")[-2] + "_" + img_2[i].split("/")[-2] + "_cm"

        in_path = img_1[i].split("sub")[0]
        path_new = os.path.join(in_path, out_path)
        os.makedirs(path_new, exist_ok=True)

        img_path = os.path.join(path_new, out_path + "_" + img_ext)
        lbl_path = os.path.join(path_new, out_path + "_" + lesion_ext)

        sitk.WriteImage(img, img_path)
        sitk.WriteImage(lbl, lbl_path)


# Generate augmentation list on the basis of voxel size and bin_id
def gen_aug_list(df, bin_id, min_voxels, max_voxels, num_samples=-1):
    df = df.loc[df["Bin_Id"] == bin_id]

    X = df["X"].tolist()
    Y = df["Y"].tolist()
    voxel = df["Voxel_Count"].tolist()

    if num_samples == -1:
        aug_list = [{'X_1': X[i], "X_2": X[j], "Y_1": Y[i], "Y_2": Y[j]}
                    for i in range(len(X))
                    for j in range(len(X))
                    if (i != j and voxel[i] + voxel[j] <= max_voxels)]
    else:
        pairs, aug_list = [], []
        for i in range(num_samples):
            coord_1, coord_2 = get_random_index(0, num_samples)

            while (coord_1, coord_2) in pairs or voxel[coord_1] + voxel[coord_2] < min_voxels or voxel[coord_1] + voxel[coord_2] > max_voxels:
                coord_1, coord_2 = get_random_index(0, num_samples)

            pairs.append((coord_1, coord_2))
            aug_list.append({'X_1': X[coord_1], "X_2": X[coord_2], "Y_1": Y[coord_1], "Y_2": Y[coord_2]})

    pd.DataFrame(aug_list).to_csv("aug_list.csv", index=False)
    return "aug_list.csv"


if __name__ == "__main__":
    ordered_set = pd.read_csv("ordered_set.csv")
    targeted_augmentation(gen_aug_list(ordered_set, 4, 100000, 10000000, 60))
    targeted_augmentation("aug_list.csv")
