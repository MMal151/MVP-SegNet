from src.utils.CommonUtils import str_to_tuple
from src.utils.ConfigurationUtils import MODEL_CFG, get_configurations
from src.utils.ProcessUtils import get_filters

CLASS_NAME = "[Model/Model]"


class Model:
    def __init__(self, input_shape):
        lgr = CLASS_NAME + "[init()]"
        self.input_shape = (*str_to_tuple(input_shape), 1)
        self.cfg = get_configurations(MODEL_CFG)

        self.conv_mode = self.cfg["conv_mode"]
        self.output_classes = self.cfg["output_classes"]
        self.dropout = self.cfg["dropout"]
        self.filters = get_filters(self.cfg["min_filter"], 5)

    def generate_mode(self):
        pass

    def print_info(self, lgr):
        print(f"{lgr}: Model initialized using the following configurations.")
        print(self.__dict__)
