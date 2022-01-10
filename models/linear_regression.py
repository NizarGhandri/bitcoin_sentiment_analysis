from models.model import BaseModel
import statsmodels.api as sm



class LinearRegression(BaseModel): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.pre_model = sm.OLS(*self.prep_data())



    def prep_data(self):
        return self.generator.y, sm.add_constant(self.generator.X)
     
        

    def train(self):
        self.model = self.pre_model.fit(cov_type="HC0")
        print(self.model.summary())


    
    def predict (self, X, online=True):
        return self.model.predict(sm.add_constant(X)).values





     



    
