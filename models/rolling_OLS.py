from models.model import Model
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

class RollingLinearRegression(Model): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.pre_model = RollingOLS(*self.prep_data(), window=len(self.generator.X))



    def prep_data(self):
        X = sm.add_constant(self.generator.data[self.generator.features])
        y = self.generator.data["log_returns"]
        return y, X
     
        

    def train(self):
        self.model = self.pre_model.fit(cov_type = "HC0")
        self.params = self.model.params.fillna(method="bfill")



    
    def predict (self, X):
        preds = []
        X = sm.add_constant(X)
        for i in X.index:
            preds.append(X.loc[i].values @ self.params.loc[max(i-1, 0)].values.T)
        return preds

