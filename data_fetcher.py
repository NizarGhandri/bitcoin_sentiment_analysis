from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import pandas as pd
import yfinance as yf
from tensorflow.keras.utils import Sequence

from preprocessing import Factors, TweetPreprocessor


class TweetGenerator(Sequence): 

    def __init__(self, config, pre_process=True):
        self.cfg = config
        self.pre_process = pre_process
        self.preprocessor = TweetPreprocessor(self.cfg)
        self.data = self._load_data()
        self._size = self.cfg.periods * self.cfg.timestep_size
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self._size//self.cfg.batch_size + 1

    def __getitem__(self, index):
        tmp_data = self.data.iloc[index]
        return tmp_data if self.pre_process else self.preprocess(tmp_data)


    def _load_data(self): 
        data = pd.read_csv(self.cfg.data_path, infer_datetime_format=True)
        data = data.dropna(subset=['text']).reset_index(drop=True)
        return self.preprocess(data) if self.pre_process else data


    def preprocess(self, x):
        return self.preprocessor(x)
    

    


class MarketAndPriceGenerator(Sequence):

    def __init__ (self, generate, cfg):
        self.cfg = cfg
        self.factors_generator = Factors()
        self.generate = generate
        self._size = self.cfg.periods * self.cfg.timestep_size
        self.data = self._load_data()


    def __len__(self):
        return self._size//self.cfg.batch_size + 1


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