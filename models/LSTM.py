from models.model import Model
from keras.models import Model
from keras.layers import Concatenate, LSTM, Input, concatenate, LeakyReLU, Dense
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import DataMode




class LSTM(Model): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.build_nn()



    def prep_data(self):
        pass
     
        

    def train(self):
        self.model.fit(x=self.generator, validation_split=0.15, epochs=100, callbacks=self.callbacks())
        


    
    def predict (self, X):
        self.generator.mode = DataMode.TESTING
        preds = []
        for x, y in self.generator:
            preds.append(self.model.predict(x))
            self.model.train_on_batch(x, y) 

        return preds


    def build_nn(self):
        
        print("building")
        sentiment = Input(shape=(self.cfg.past_values, 1))
        first_LSTM = LSTM(1, )(sentiment)

        ret = Input(shape=(self.cfg.past_values, 1))
        second_LSTM = LSTM(1, )(ret)

        rsi = Input(shape=(self.cfg.past_values, 1))
        third_LSTM = LSTM(1, )(rsi)

        macd = Input(shape=(self.cfg.past_values, 1))
        fourth_LSTM = LSTM(1, )(macd)

        merge_one = concatenate([first_LSTM, second_LSTM, third_LSTM, fourth_LSTM])

        std = Input(shape=(1, ))
        merge_two = concatenate([merge_one, std])


        hidden_layer = LeakyReLU()(Dense(7, )(merge_two,))

        output_layer = Dense(1,)(hidden_layer)

        

        self.model = Model(inputs=[std, sentiment, ret, macd, rsi], outputs=output_layer)
        ada_grad = adam_v2(lr=0.0001)
        self.model.compile(optimizer=ada_grad, loss='mean_squared_error',
                    metrics=['root_mean_squared_error'])
        print(self.model.summary())



    def callbacks(self):
        """
        function to get checkpointer, early stopper and lr_reducer in our CNN
        """

        #Stop training when f1_m metric has stopped improving for 10 epochs
        earlystopper = EarlyStopping(monitor = "val_root_mean_squared_error", 
                                    mode='min', 
                                    patience = 20,
                                    verbose = 1,
                                    restore_best_weights = True)

        #Reduce learning rate when loss has stopped improving for 5 epochs
        lr_reducer = ReduceLROnPlateau(monitor='loss',
                                    mode='min',
                                    factor=0.8,
                                    patience=5,
                                    min_delta= 0.001, 
                                    min_lr=0.00001,
                                    verbose=1)

        return [earlystopper, lr_reducer]
