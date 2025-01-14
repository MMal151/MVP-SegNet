import nibabel as nib
import pandas as pd

from src.dataset.Dataset import Dataset
from src.utils.CommonUtils import save_csv
from src.utils.DataPrepUtils import non_empty_patch

CLASS_NAME = "[DataPreparation/ImageDataset]"


class ImageDataset(Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_of_images = cfg["num_images"]  # Number of images to be extracted from each scan.
        self.alw_empty_images = cfg["alw_empty_images"]
        self.images = None

    def generate_splits(self):
        super().generate_splits()
        self.images = self.generate_images()
        self.save_csv()

    def generate_images(self):
        lgr = CLASS_NAME + "[generate_images()]"
        train_images, valid_images, test_images = [], [], []
        img_dict = {}

        print(f"{lgr}: Dividing Training Images.")

        for i in self.train_y:
            train_images.append(extract_imgs(i, self.num_of_images, self.alw_empty_images))

        img_dict["train"] = train_images

        print(f"{lgr}: Dividing Validation Images.")

        for i in self.valid_y:
            valid_images.append(extract_imgs(i, self.num_of_images, True))
        img_dict["valid"] = valid_images

        for i in self.test_y:
            test_images.append(extract_imgs(i, self.num_of_images, True))
        img_dict["test"] = test_images

        return img_dict

    def save_csv(self):
        # Using 'patches' column to save the images coordinates to ensure that the behavior is seamless with the 3D-Dataset Generator.

        save_csv("DataFiles/", "train.csv",
                 pd.DataFrame({'X': self.train_x, 'Y': self.train_y, 'patches': self.images["train"]}))
        save_csv("DataFiles/", "valid.csv",
                 pd.DataFrame({'X': self.valid_x, 'Y': self.valid_y, 'patches': self.images["valid"]}))
        save_csv("DataFiles/", "test.csv",
                 pd.DataFrame({'X': self.test_x, 'Y': self.test_y, 'patches': self.images["test"]}))


def extract_imgs(img, num_of_images=-1, alw_empty_images=False):
    imgs = []
    msk = nib.load(img).get_fdata()

    if num_of_images <= 0:
        stride = 1
    else:
        stride = abs(msk.shape[2] / num_of_images)

    if alw_empty_images:
        imgs = [(0, 0, j) for j in range(0, msk.shape[2], stride)]
    else:
        imgs = [(0, 0, j) for j in range(0, msk.shape[2], stride) if non_empty_patch(msk[:, :, j])]

    return imgs

