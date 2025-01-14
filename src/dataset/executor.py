from src.dataset.BalancedDataset import BalancedDataset
from src.dataset.ImageDataset import ImageDataset
from src.dataset.NiftiDataset import NiftiDataset
from src.utils.ConfigurationUtils import DATAPREP_CFG, get_configurations

CLASS_NAME = "[DataPreparation/executor]"


def prepare_dataset(cfg=None):
    if cfg is None:
        cfg = get_configurations(DATAPREP_CFG)
    lgr = CLASS_NAME + "[execute()]"

    # Generate train/test/valid Splits
    if cfg["mode"].lower() == "gen_splits":
        if cfg["split"]["type"].lower() == "balanced":
            BalancedDataset(cfg["split"]).generate_splits()
        else:
            NiftiDataset(cfg["split"]).generate_splits()
    elif cfg["mode"].lower() == "gen_images":
        ImageDataset(cfg["2D"]).generate_splits()


if __name__ == "__main__":
    prepare_dataset()
