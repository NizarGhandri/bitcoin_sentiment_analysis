
from models.model import BaseModel
from keras.models import Model
from keras.layers import LSTM, Input, concatenate, LeakyReLU, Dense, Dropout, ReLU
from keras.activations import tanh
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import Huber
from metrics import Metrics
import numpy as np



class LSTM_nn(BaseModel): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.arch = kwargs["arch_type"]
        if(self.arch == 1):
            self.build_nn()
        else:
            self.build_nn_2()



    def prep_data(self, X=None):
        df = self.generator.X if X is None else X
        std = np.array(df["sentiment_score_std"])
        sentiment =  df[["sentiment_score_mean_n_1", "sentiment_score_mean_n_2", 'sentiment_score_mean_n_3']].values[..., np.newaxis].astype('float32')
        ret =  df[["log_returns_n_1", 'log_returns_n_2', 'log_returns_n_3']].values[..., np.newaxis].astype('float32')
        macd = df[['signal_macd_n_1', 'signal_macd_n_2', 'signal_macd_n_3']].values[..., np.newaxis].astype('float32')
        rsi = df[['rsi_n_1', 'rsi_n_2', 'rsi_n_3']].values[..., np.newaxis].astype('float32')
        #print(sentiment.shape)
        #print(np.concatenate([sentiment, ret, macd, rsi], axis=2).shape)
        final = [std, sentiment, ret, macd, rsi] if (self.arch == 1) else [std, np.concatenate([sentiment, ret, macd, rsi], axis=2)]
        return [std, np.concatenate([sentiment, ret, macd, rsi], axis=2)], self.generator.y

     
        

    def train(self):
        X, y = self.prep_data()
        X_train, y_train = list(map(lambda x: x[:-10], X)), y[:-10]
        X_val, y_val = list(map(lambda x: x[-10:], X)), y[-10:]
        self.model.fit(X_train, y_train , batch_size=self.cfg.batch_size, validation_data=(X_val, y_val), epochs=1000, callbacks=self.callbacks())
        


    
    def predict (self, X):
        #self.generator.mode = DataMode.TESTING
        #preds = []
        #for x, y in self.generator:
        #    preds.append(self.model.predict(x))
        #    self.model.train_on_batch(x, y) 
        pred, _ = self.prep_data(X)
        return self.model.predict(pred)


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


        hidden_layers = Dense(8, )(merge_two,)

        hidden_layers = tanh(hidden_layers)

        hidden_layers = Dropout(0.5)(hidden_layers)

        output_layer = Dense(1,)(hidden_layers)


        

        self.model = Model(inputs=[std, sentiment, ret, macd, rsi], outputs=output_layer)
        self.model.compile(optimizer="adam", loss="mean_absolute_error",
                    metrics=Metrics._get_all())
        print(self.model.summary())


    def build_nn_2(self):
        
        print("building")
        sentiment = Input(shape=(self.cfg.past_values, 4))
        first_LSTM = LSTM(1, )(sentiment)

        std = Input(shape=(1, ))
        merge_two = concatenate([first_LSTM, std])


        hidden_layers = Dense(7, )(merge_two,)

        hidden_layers = ReLU()(hidden_layers)

        hidden_layers = Dropout(0.5)(hidden_layers)

        output_layer = Dense(1,)(hidden_layers)


        

        self.model = Model(inputs=[std, sentiment], outputs=output_layer)
        self.model.compile(optimizer="adam", loss="mean_absolute_error",
                    metrics=Metrics._get_all())
        print(self.model.summary())



    def callbacks(self):
        """
        function to get checkpointer, early stopper and lr_reducer in our CNN
        """

        #Stop training when f1_m metric has stopped improving for 10 epochs
        earlystopper = EarlyStopping(monitor = "val_loss", 
                                    mode='min', 
                                    patience = 10,
                                    verbose = 1,
                                    restore_best_weights = True)

        #Reduce learning rate when loss has stopped improving for 5 epochs
        lr_reducer = ReduceLROnPlateau(monitor='loss',
                                    mode='min',
                                    factor=0.5,
                                    patience=5,
                                    min_delta= 0.001, 
                                    min_lr=0.000001,
                                    verbose=1)

        return [earlystopper, lr_reducer]
