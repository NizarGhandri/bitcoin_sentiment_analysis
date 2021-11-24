from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import got3
import yfinance as yf


class DataFetcher(object): 

    def __init__(self, config, override=False, pre_process=True, max_workers=8):
        self.BEARER_TOKEN = BEARER_TOKEN
        self.cfg = config
        self.pre_process = pre_process
        self.max_workers = max_workers
        self.override = override
        self.data = None
        self.fetched = 0
        



    def __call__ (self):
        with ThreadPoolExecutor(max_workers = self.max_workers) as executor:
            results = executor.map(square, self.cfg.periods)

        return results
         

    def fetch(self, period):
        ### Tweets ###
        tweetCriteria = got3.manager.TweetCriteria().setQuerySearch(self.cfg.stock_name).setSince(period.low).setUntil(period.high).setMaxTweets(self.cfg.max_tweets_per_worker)
	    tweets = got3.manager.TweetManager.getTweets(tweetCriteria)

        ### Prices ###
        stock_ticker = yf.Ticker(self.cfg.stock)
        stock_history = stock_ticker.history(
            start=period.low,
            end=period.high,
            interval=self.cfg.ticker_interval
        ).reset_index()

        return tweets, stock_history
        
    
    def pre_process (self): 
        pass


    