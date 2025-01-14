import nibabel as nib
import pandas as pd

from src.utils.DataPrepUtils import get_nonempty_patches, get_patch_coordinates_3D
from src.dataset.Dataset import Dataset
from src.utils.ProcessUtils import augmentation_cm
from src.utils.CommonUtils import save_csv, str_to_tuple

CLASS_NAME = "[DataPreparation/Dataset]"


class NiftiDataset(Dataset):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.do_patching = cfg["do_patching"]

        if cfg["do_patching"]:
            self.patch_shape = str_to_tuple(cfg["patch"]["shape"])
            self.random_patches = cfg["patch"]["random"]
            self.stride = cfg["patch"]["stride"]
            self.alw_empty_patches = cfg["patch"]["alw_empty_patches"]
            self.patch_coords = None

        self.do_augmentation = cfg["do_augmentation"]

        if cfg["do_augmentation"]:
            self.augmentation_factor = cfg["augmentation"]["factor"]

    def generate_splits(self):
        super().generate_splits()

        if self.do_augmentation:
            self.train_x, self.train_y = augmentation_cm(self.train_x, self.train_y, self.x_ext, self.y_ext, self.augmentation_factor)

        if self.do_patching:
            if not self.random_patches:
                self.patch_coords = self.generate_ordered_patches()

        self.save_csv()

    def generate_ordered_patches(self):
        lgr = CLASS_NAME + "[generate_ordered_patches()]"
        patch_coords = {}  # List of patch coordinates w.r.t to image shape.
        train_patches, valid_patches, test_patches = [], [], []
        patches_dict = {}

        print(f"{lgr}: Dividing Training Patches. ")

        for i in self.train_y:
            msk = nib.load(i)
            if not (*msk.shape, self.stride["train"]) in patch_coords.keys():
                patch_coords[(*msk.shape, self.stride["train"])] = get_patch_coordinates_3D(msk.shape, self.stride["train"], self.patch_shape)
            patches = patch_coords[(*msk.shape, self.stride["train"])]
            if not self.alw_empty_patches:
                patches = get_nonempty_patches(msk, patches, self.patch_shape)

            train_patches.append(patches)

        patches_dict['train'] = train_patches

        print(f"{lgr}: Dividing Validation Patches. ")

        if self.valid_y is not None:
            for i in self.valid_y:
                msk = nib.load(i)
                if not (*msk.shape, self.stride["valid"]) in patch_coords.keys():
                    patch_coords[(*msk.shape, self.stride["valid"])] = get_patch_coordinates_3D(msk.shape, self.stride["valid"], self.patch_shape)
                valid_patches.append(patch_coords[(*msk.shape, self.stride["valid"])])

        patches_dict['valid'] = valid_patches

        print(f"{lgr}: Dividing Testing Patches. ")

        if self.test_y is not None:
            for i in self.test_y:
                msk = nib.load(i)
                if not (*msk.shape, self.stride["test"]) in patch_coords.keys():
                    patch_coords[(*msk.shape, self.stride["test"])] = get_patch_coordinates_3D(msk.shape, self.stride["test"], self.patch_shape)
                test_patches.append(patch_coords[(*msk.shape, self.stride["test"])])

        patches_dict['test'] = test_patches

        return patches_dict

    def save_csv(self):
        if self.do_patching:
            save_csv("DataFiles/", "train.csv",
                     pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': self.patch_coords['train']}))
            save_csv("DataFiles/", "valid.csv",
                     pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': self.patch_coords['valid']}))
            save_csv("DataFiles/", "test.csv",
                     pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': self.patch_coords['test']}))
        else:
            save_csv("DataFiles/", "train.csv",
                     pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': None}))
            save_csv("DataFiles/", "valid.csv",
                     pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': None}))
            save_csv("DataFiles/", "test.csv",
                     pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': None}))
