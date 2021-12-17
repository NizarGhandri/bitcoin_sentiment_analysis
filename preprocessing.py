import numpy as np
import re
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
#import dask.dataframe as dd
#from dask.multiprocessing import get
from torch.utils.data import Dataset, DataLoader
from transformers.utils.dummy_pt_objects import EncoderDecoderModel
from pytrends.request import TrendReqs


class FastSentimentData(Dataset):

    def __init__ (self, x): 
        self.x = x

    def __len__(self): 
        return len(self.x)

    def __getitem__ (self, idx): 
        return self.x[idx]

class TweetPreprocessor(object): 

    
    def __init__(self, _cfg) -> None:
        self._cfg = _cfg
        self.preprocessing = list(map(lambda f: getattr(self, f), filter(lambda method: not method.startswith('_'), dir(self))))
        self.aggregation = {'sentiment_score': ["mean", "std"]}
    
    
    
    def __call__(self, x):
        for func in self.preprocessing: 
            x = func(x).reset_index(drop=True)
        
        _x = x.loc[:, ["date", "sentiment_score", "period"]].groupby("period")

        #if (not os.path.exists(self.cfg.preprocessed_path)):
        #    os.mkdir(self.cfg.preprocessed_path)
        
        #for name, dataframe in _x.__iter__():
        #    dataframe.to_csv(os.path.join(self.cfg.preprocessed_path, "preprocessed_"+str(name)+".csv"))
        
        _x = _x.apply(lambda df: self._time_travel(df.groupby(pd.Grouper(key="date", freq=self._cfg.ticker_interval)).agg(self.aggregation).reset_index()))


        return _x

    
    def _time_travel(self, df):
        #print(df.columns)
        new_df = df.iloc[self._cfg.past_values:].copy()
        for i in range(1, self._cfg.past_values+1):
            #print(df.iloc[(3-i):-i].loc[:, ('sentiment_score', 'mean')].tolist()[:3])
            new_df.loc[:, ('sentiment_score', 'mean_n_'+str(i))] =  df.iloc[(self._cfg.past_values-i):-i].loc[:, ('sentiment_score', 'mean')].tolist()
        return new_df.reset_index(drop=True)


    def filter_in_periods(self, x):
        x.loc[:,"date"] = pd.to_datetime(x["date"], errors="coerce", infer_datetime_format=True)
        x.loc[:,"period"] = x.date.apply(self._in_period)
        return x.dropna(subset=['period', 'date'])


    def remove_bots(self, x):
        x.loc[:, "source"]= x["source"].apply(str).str.lower()
        x = x[~x["source"].str.contains("bot")]
        return x

    def preprocess_tweet_text(self, x):
        x.loc[:, "text"] = x["text"].apply(str).apply(self._process_tweet)
        return x

    #def numeric_verified(x):
    #    x['user_verified'] = x['user_verified'].apply(lambda u: int(u =='True'))

    def _process_tweet(self, tweet): #start process_tweet
        # process the tweets
        #Convert to lower case
        tweet = tweet.lower()
        #Convert www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
        #Convert @username to AT_USER
        tweet = re.sub('@[^\s]+','AT_USER',tweet)
        #Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)#trim
        tweet = tweet.strip('\'"')
        return tweet

    #def summon_bert(self, x):
    #    pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=0, return_all_scores=False)
    #    partitioned_dask = dd.from_pandas(x.text, npartitions=30)
    #    x["sentiment_score"] = partitioned_dask.map_partitions(lambda df: df.apply(pipe)).compute(scheduler='threads')
    #    return x

    def _get_score(self, score):
        #print(score) 
        return score['score'] * (2*(score["label"] == "POSITIVE") - 1)

    def summon_bert(self, x):
        print("we berting")
        pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-(not self._cfg.cuda), return_all_scores=False)
        #partitioned_dask = dd.from_pandas(x.text, npartitions=30) .apply(pipe).apply(_get_score)
        it = x.text.tolist()
        #n = len(it)
        #batch_size = 256
        loader = DataLoader(FastSentimentData(it), batch_size=256, pin_memory=True, num_workers=16)#[pipe(it[x:min(x+batch_size, n)]) for x in tqdm(range(0, n, batch_size))]
        x["sentiment_score"] = [elem for data in tqdm(loader) for elem in pipe(data)]#[self._get_score(item) for batch in preds for item in batch]
        x["sentiment_score"] = x["sentiment_score"].apply(self._get_score)
        x[["text", "sentiment_score"]].to_csv("test.csv")
        return x    

        
    def _in_period(self, dt):
        period = list(filter(lambda p: p[1].contains(dt), enumerate(self._cfg.periods))) 
        return period[0][0] if len(period) else np.NaN



        

class Factors(object):


    def __init__(self):
        pytrend = TrendReq()

    def __call__(x):
        return x


    def MACD_and_signal_MACD(self, x):
        exp1 = x["price"].ewm(span=12, adjust=False).mean()
        exp2 = x["price"].ewm(span=26, adjust=False).mean()
        x["macd"] = exp1 - exp2
        x["signal_macd"] = x["macd"].ewm(span=9, adjust=False).mean()
        return x

    

    def RSI(self, x):
        diff =  x["price"].diff(1)
        gain =  diff.clip(lower=0).round(2)
        loss =  diff.clip(upper=0).abs().round(2)
        rs = gain.rolling(window=self.window_length, min_periods=self.window_length).mean()/loss.rolling(window=self.window_length, min_periods=self.window_length).mean()#[:self.window_length+1]
        x['rsi'] = 1.0 - (1.0 / (1.0 + rs))
        return x

    
    def bollinger_bands(self, x):
        avg = x["price"].rolling(window=self.window_length, min_periods=self.window_length).mean()
        std = x["price"].rolling(window=self.window_length, min_periods=self.window_length).std()
        x["Bollinger_low"], x["Bollinger_high"] = avg - 1.96*std, avg + 1.96*std
        return x

    
    def OBV(self, x):
        return np.where(x['close'] > x['close'].shift(1), x['volume'], np.where(x['close'] < x['close'].shift(1), -x['volume'], 0)).cumsum()

    def google_trends(self, x):
        keyw = pytrend.suggestions(keyword=self.cfg.stock_name)["title"]
        df = pytrend.build_payload(kw_list=keyw, startTime=self.cfg.periods[0].low, endTime=self.cfg.periods[-1].high)
        x["google trends"] = df.groupby(pd.Grouper(freq=self.cfg.ticker_interval)).sum().diff(1) 
        return x

    

    



    #Google Search Interest,
    #Active supply Bitcoin