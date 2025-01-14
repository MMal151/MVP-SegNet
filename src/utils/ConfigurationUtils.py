import yaml
import os

# -- Configuration Files -- #
MODEL_CFG = os.path.dirname(__file__) + "//..//ConfigurationFiles//ModelConfigurations.yml"  # Model Configuration File
DATAPREP_CFG = os.path.dirname(__file__) + "//..//ConfigurationFiles//DataPrepConfig.yml"  # Data Preparation Configuration File
LR_CFG = os.path.dirname(__file__) + "//..//ConfigurationFiles//LearningSchedularConfig.yml"  # Learning Rate Schedular Configuration File


def get_configurations(config_file="_config.yml"):
    with open(config_file, "r") as configFile:
        return yaml.safe_load(configFile)
