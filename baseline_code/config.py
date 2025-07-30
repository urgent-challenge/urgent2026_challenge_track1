import os
import yaml
import argparse

class Config():

    
    def __init__(self, **kwargs):
        self.learning_rate = 1e-3
        self.batch_size = 2
        self.weight_decay = 1e-6
        self.adam_epsilon = 1e-8
        self.num_worker = 4
        self.num_train_epochs = 150
        self.device = "cuda"
        self.num_gpu = 1
        self.train_version = 0
        self.train_tag = "run_0"
        self.train_name = 'baseline'
        self.val_check_interval = 50000
        self.save_top_k = 3
        self.resume = True
        self.seed = 1996
        self.gradient_clip = 0.5
        self.lr_step_size = 1
        self.lr_gamma = 0.85
        self.train_set_path = 'none'
        self.train_set_dynamic_mixing = True
        self.valid_set_path = 'none'
        self.init_from = 'none'
        self.max_duration=96000
        self.use_high_pass = True
        self.se_model='bsrnn'
        self.config_file = "none"
        self.model_configs = None

        for k, v in kwargs.items():
            self.__setattr__(k, v)
    

    def read_yaml(self):
        if self.config_file != 'none':

            file = open(self.config_file, 'r', encoding='utf-8')
            string = file.read()
            dict = yaml.safe_load(string)
            for k, v in dict.items():
                self.__setattr__(k, v)
            
            self.train_tag = os.path.basename(self.config_file).replace('.yaml', '')

        return
    
def config_parser():
    cfg = Config()
    parameters = vars(cfg)

    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    for par in parameters:
        default = parameters[par]
        parser.add_argument(f"--{par}", type=str2bool if isinstance(default, bool) else type(default), default=default)
    args = parser.parse_args()
    return args