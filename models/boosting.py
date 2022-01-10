import numpy as np
from models.model import BaseModel
from tqdm.autonotebook import tqdm
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class Boosting(BaseModel): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.params = self.cfg.params
        self.build_model()



    def prep_data(self, X):
        pass

    def build_model(self):
        self.model = XGBRegressor(booster = "gbtree", **self.params)
        

    def train(self, cv=False):
        if cv:
            res, idx = [], []
            for lmbd in tqdm(self.cfg.l):
                for d in tqdm(range(1,9), leave=False):
                    for e in tqdm(range(10, 110, 10), leave=False):
                        
                        model = XGBRegressor(booster = "gbtree", reg_lambda=lmbd, max_depth=d, n_estimators=e)

                        kfold = KFold(n_splits=self.cfg.splits)
                        results = cross_val_score(model, self.generator.X, self.generator.y, cv=kfold, scoring="neg_mean_absolute_error")

                        res.append(results.copy())
                        idx.append((lmbd, d, e))
            
            self.params = dict(zip(["reg_lambda", "max_depth", "n_estimators" ], idx[np.argmax(np.median(np.array(res), axis=1))]))
            self.build_model()

        self.model.fit(self.generator.X, self.generator.y)
        


    
    def predict (self, X, y=None, online=False):
        if online:
            assert(online and not (y is None)), "if online provide a y"
            preds = []
            data = self.generator.X.values.copy()
            labels = self.generator.y.values.copy()
            for x, y in zip(X.values, y.values):
                x = x[np.newaxis, ...]
                preds.append(model.predict(x))
                data, labels = np.concatenate([data, x]), np.append(labels, y)
                print(data.shape)
                model = XGBRegressor(booster = "gbtree", reg_lambda=0.005, max_depth=1, n_estimators=30)
                model.fit(data, labels)
            return np.concatenate(preds)
        else:
            return self.model.predict(X)


        
        

    
    