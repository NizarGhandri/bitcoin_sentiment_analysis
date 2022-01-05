""" Contains configuration attributes used during preprocessing and training. """
from re import X
import numpy as np
from multiprocessing import cpu_count
from tensorflow.keras import  regularizers
import os
from datetime import datetime
from torch.cuda import is_available


class Period:
    def __init__(self, low, high, dt_format="%Y-%m-%d"): 
        self.low = datetime.strptime(low, dt_format)
        self.high = datetime.strptime(high, dt_format)

    def contains(self, x):
        return x >= self.low and  x <= self.high  




class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
    cfg = Config()
    
    """ 
    **************************************** Paths **************************************** 
    """
    cfg.data_path = os.path.join("data", "Bitcoin_tweets.csv")
    cfg.preprocessed_path = os.path.join("data", "preprocessed")
    cfg.save_path = os.path.join("models", "saved_models")
    """ 
    ************************************************************************************************
    """ 


    cfg.cuda = is_available()
    cfg.device =  "cuda" if cfg.cuda else "cpu"

    """ Data Fetcher """

    intervals = [("2021-02-01", "2021-10-24"), ("2021-10-25", "2021-11-07"), ("2021-11-08", "2021-11-26")] 
    cfg.periods = list(map(lambda x: Period(*x), intervals))
    cfg.keyw = ["Bitcoin"]
    cfg.max_tweets_per_worker = 1000
    cfg.stock = "BTC-USD"
    cfg.ticker_interval = '12h'
    cfg.timestep_size = 2
    cfg.past_values = 3
    cfg.policy = "Interpolate"

    """ Dataset """  
    # rescaling parameters
    cfg.label_idx = -6
    cfg.shuffle = True

    
    # input should be cubic. Otherwise, input should be padded accordingly.
    #cfg.patch_shape = (32, 32, 32)

    # preprocessing config - options are:
    cfg.preprocess = True 
    cfg.max_len = 72000
    cfg.sample_rate = 16000 

    """ Augmentations """

    cfg.augmentation_rate = 16
    cfg.shifts = 5
    cfg.time_shift_bound = cfg.sample_rate//2
    cfg.pitch_shift_bound = 12
    cfg.mean = 0
    cfg.std = 0.1

    """ feature engineering"""
    
    cfg.n_mfcc=40
    cfg.hop_length=1024

    """loader"""
    cfg.loader_cores = cpu_count()

    """ Model """
    cfg.input_shape = (cfg.n_mfcc, cfg.max_len//cfg.hop_length + 1, 1)
    cfg.batch_size = 32
    cfg.epochs = 1


    cfg.conv_layer_count = 3
    cfg.dense_layer_count = 1

    cfg.n_filters = [16, 24, 32]
    cfg.n_kernels = [(4,8), (2,4), (2,2)]
    cfg.n_maxpooling = [(3,3), (2,2), (3,3)]
    cfg.n_dense = [64]
    cfg.n_classes = 7


    cfg.dropouts_conv = [0.2]*cfg.conv_layer_count
    cfg.dropouts_dense = [0.5]*cfg.dense_layer_count
    cfg.regularizers_conv = [regularizers.l1_l2(l1=1e-5, l2=1e-4)]*cfg.conv_layer_count
    cfg.regularizers_dense = [regularizers.l1_l2(l1=1e-4, l2=1e-4)]*cfg.dense_layer_count
    cfg.batch_norm = True  

    cfg.activation_Leaky = True
    cfg.alpha = 0.3
    cfg.last_activation = "softmax"
    cfg.loss = "binary_crossentropy"
    
    """ Optimizer """
    cfg.optimizer = "adam"

    cfg.translator = {"W": "anger", "L":"boredom", "E": "disgust", "A":"anxiety/fear", "F":"happiness", "T":"Trauer", "N":"neutral"}
    # """ Reporting """
    # cfg.wab = True # use weight and biases for reporting
    
    return cfg