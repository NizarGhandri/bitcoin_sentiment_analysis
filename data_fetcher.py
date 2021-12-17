from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import pandas as pd
import yfinance as yf
from tensorflow.keras.utils import Sequence
import os

from preprocessing import Factors, TweetPreprocessor


class TweetGenerator(Sequence): 

    def __init__(self, mode, config, pre_process=True, return_all_features=False, name_preprocessed="preprocessed.csv"):
        self.cfg = config
        self.mode = mode
        self.pre_process = pre_process
        self.path_preprocessed = os.path.join(self.cfg.preprocessed_path, name_preprocessed)
        self.preprocessor = TweetPreprocessor(self.cfg)
        self.data = self._load_data()
        self.return_all = return_all_features
        if(pre_process):
            self.save()
        
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data.loc[self.mode.value])//self.cfg.batch_size + 1

    def __getitem__(self, index):
        data = self.data.loc[self.mode.value, :] if self.return_all else self.data.loc[self.mode.value, "sentiment_score"]
        return data.iloc[index]


    def _load_data(self):
        if (self.pre_process): 
            data = pd.read_csv(self.cfg.data_path, infer_datetime_format=True)
            data = self.preprocess(data.dropna(subset=['text']).reset_index(drop=True))
        else:
            data = pd.read_csv(self.path_preprocessed, infer_datetime_format=True, index_col=[0, 1], header = [0,1])
            data = self.missing_value_policy(data.rename(columns = {'Unnamed: 2_level_1':  ''}, level=1))
        return data


    def preprocess(self, x):
        return self.preprocessor(x)

    def save(self):
        if (not os.path.exists(self.cfg.preprocessed_path)):
            os.mkdir(self.cfg.preprocessed_path)
        
        self.data.to_csv(self.path_preprocessed)
        
        return self.path_preprocessed

    def missing_value_policy(self, df):
        if self.cfg.policy == "Interpolate":    
            for i in range(2, self.cfg.past_values+1):
                df.loc[:, ("sentiment_score", "mean_n_"+str(i))] = df.loc[:, ("sentiment_score", "mean_n_"+str(i))].fillna(method="bfill", limit=self.cfg.past_values-1)

        return df.dropna()



    

    


class ReturnFactorsGenerator(Sequence):

    def __init__ (self, cfg, generate):
        self.cfg = cfg
        self.factors_generator = Factors()
        self.generate = generate
        self.data = self._load_data()


    def __len__(self):
        return len(self.data)//self.cfg.batch_size + 1


    def __getitem__(self, index):
        tmp_data = self.data.iloc[index]
        return tmp_data if self.generate else self.generate_factors(tmp_data)


    def _load_data(self, ):
        stock_ticker = yf.Ticker(self.cfg.stock)
        stock_history = stock_ticker.history(
            start=self.period.low,
            end=self.cfg.period.high,
            interval=self.cfg.ticker_interval
        ).reset_index()
        return self.generate_factors(stock_history) if self.generate else stock_history


    def generate_factors(self, x):
        return self.factors_generator(x) 
   


class Dataset(Sequence):

    def __init__(self, config, max_workers=8, pre_process_on_the_fly=False):
        self.cfg = config
        self.on_the_fly = pre_process_on_the_fly
        self.tweets = TweetGenerator(self.cfg, not pre_process_on_the_fly)
        self.prices_marketindicators = MarketAndPriceGenerator(self.cfg, not pre_process_on_the_fly)
        self.max_workers = max_workers
        self._size = len(TweetGenerator)

        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._size

    def __getitem__(self, index):
        if self.on_the_fly: 
            with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
                data = executor.map(self.fetch, self.cfg.periods)
        else:
            data = pd.concat(list(map(self.fetch, self.cfg.periods)))
        return data

    def fetch(self):
        pass



#def fetch(self, period):
    ### Tweets ###
#    tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(self.cfg.stock_name).setSince(period.low).setUntil(period.high).setMaxTweets(self.cfg.max_tweets_per_worker)
#    tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

    ### Prices ###
#    stock_ticker = yf.Ticker(self.cfg.stock)
#    stock_history = stock_ticker.history(
#        start=period.low,
#        end=period.high,
#        interval=self.cfg.ticker_interval
#    ).reset_index()

#    return tweets, stock_history        

#def pre_process (self): 
#   pass




#def __call__ (self):
#    with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
#        results = executor.map(self.fetch, self.cfg.periods)
#
#    return results