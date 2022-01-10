from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from metrics import Metrics
import pandas as pd
import numpy as np


class BaseModel(ABC): 


    def __init__ (self, generator, cfg, **kwargs): 
        self.generator = generator
        self.cfg = cfg
        
   

    @abstractmethod
    def prep_data(self, X): 
        pass 

    @abstractmethod 
    def train (self):
        pass

    @abstractmethod
    def predict (self, X):
        pass


    def plot(self):
        fig, ax = plt.subplots(2)
        ax[0].set_title('training')
        ax[1].set_title('test')
        ax[0].plot(self.predict(self.generator.X), label="prediction")
        ax[0].plot(self.generator.y, label="ground_truth")
        ax[0].legend()
        ax[0].grid()
        ax[1].plot(self.predict(self.generator.X_test), label="prediction")
        ax[1].plot(self.generator.y_test.values, label="ground_truth")
        ax[1].legend()
        ax[1].grid()

    def evaluate(self): 
        y_pred, y_true = self.predict(self.generator.X_test), self.generator.y_test.values
        MAE, acc = np.mean(np.abs(y_pred - y_true)), np.mean((y_pred * y_true) > 0)
        return pd.DataFrame({"MAE": [MAE], "ACCURACY": [acc]}).set_index([["model"]])


    
