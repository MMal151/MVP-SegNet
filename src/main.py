import sys

from src.dataset.executor import prepare_dataset
from src.segmentation.process.Inference import inference
from src.segmentation.process.Train import train
from src.utils.CommonUtils import is_valid_file
from src.utils.ConfigurationUtils import get_configurations

CFG_FILE = "config.yml"


def default_main():
    cfg = get_configurations(CFG_FILE)

    if cfg["mode"] == "train":
        cfg = cfg["train"]
        if cfg["prepare_dataset"]:
            prepare_dataset()
        train(cfg)

    elif cfg["mode"] == "inference":
        inference(cfg)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        if args[0] == "--train":
            if args[1] == "--config":
                CFG_FILE = args[2]
                if is_valid_file(CFG_FILE):
                    cfg = get_configurations(CFG_FILE)
                    train(cfg)
                else:
                    raise Exception(f"Invalid configuration file given. Check if file exists.")
            else:
                raise Exception(f"No configuration file given.")
        elif args[0] == "--inference":
            if args[1] == "--config":
                CFG_FILE = args[2]
                if is_valid_file(CFG_FILE):
                    cfg = get_configurations(CFG_FILE)
                    inference(cfg)
                else:
                    raise Exception(f"Invalid configuration file given. Check if file exists.")
            else:
                raise Exception(f"No configuration file given.")
        elif args[0] == "--prepare_dataset":
            if args[1] == "--config":
                CFG_FILE = args[2]
                if is_valid_file(CFG_FILE):
                    cfg = get_configurations(CFG_FILE)
                    prepare_dataset(cfg)
                else:
                    raise Exception(f"Invalid configuration file given. Check if file exists.")

            else:
                raise Exception(f"No configuration file given.")
        else:
            raise Exception(
                f"Invalid value given. Expected Set of Values: [train, inference, prepare_dataset]. Given Value: [{args[0]}]")
    else:
        default_main()
