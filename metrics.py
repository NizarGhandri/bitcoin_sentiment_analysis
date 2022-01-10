from tensorflow.keras import backend as K
from keras.metrics import RootMeanSquaredError


class Metrics: 


    rmse = RootMeanSquaredError()

    @staticmethod
    def coeff_determination(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    @staticmethod
    #compute recall
    def _recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    #compute precision
    def _precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    #compute F1-score
    def _nizar(y_true, y_pred):
        precision = Metrics.correct_dir(y_true, y_pred)
        recall = 1/(K.mean(K.abs(y_true - y_pred)))
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    @staticmethod
    def correct_dir(y_true, y_pred):
        return K.mean((y_true*y_pred) > 0)


    

    @staticmethod
    def _get_all() : 
        return list(map(lambda f: getattr(Metrics, f), filter(lambda method: not method.startswith('_'), dir(Metrics))))

    @staticmethod
    def _get_dict() : 
        return dict(map(lambda f: (f, getattr(Metrics, f)), filter(lambda method: not method.startswith('_'), dir(Metrics))))