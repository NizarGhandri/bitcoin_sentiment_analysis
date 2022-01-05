from abc import ABC, abstractmethod
import matplotlib.pyplot as plt



class Model(ABC): 


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



    
