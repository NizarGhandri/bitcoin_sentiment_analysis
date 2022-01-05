from models.model import Model
import statsmodels.api as sm
import pandas as pd


class Varmax(Model): 


    def __init__ (self, generator, cfg, **kwargs): 
        super().__init__(generator, cfg, **kwargs)
        self.pre_model = sm.tsa.VARMAX(self.prep_data(), order=(3,0), exog=self.generator.y)



    def prep_data(self):
        sent = self.generator.X[["sentiment_score_mean_n_1", "sentiment_score_mean_n_2", 'sentiment_score_mean_n_3']].head(1).T.reset_index(drop=True)
        ret =  self.generator.X[["log_returns_n_1", 'log_returns_n_2', 'log_returns_n_3']].head(1).T
        macd = self.generator.X[['signal_macd_n_1', 'signal_macd_n_2', 'signal_macd_n_3']].head(1).T
        rsi = self.generator.X[['rsi_n_1', 'rsi_n_2', 'rsi_n_3']].head(1)
        tmp = pd.concat([sent, ret, macd, rsi], axis=1)
        X = self.generator.X[['sentiment_score_mean', 'sentiment_score_std', 'rsi', 'macd']]
        return self.generator.y, sm.add_constant(X)
     
        

    def train(self):
        self.model = self.pre_model.fit()
        print(self.model.summary())


    
    def predict (self, X):
        return self.model.predict(sm.add_constant(X))



mod = 
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
final_data["log_returns"].plot()
plt.plot(res.predict())