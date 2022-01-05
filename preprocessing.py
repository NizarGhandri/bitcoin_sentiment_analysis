import numpy as np
import re
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
#import dask.dataframe as dd
#from dask.multiprocessing import get
from torch.utils.data import Dataset, DataLoader
from transformers.utils.dummy_pt_objects import EncoderDecoderModel, MarianForCausalLM
from pytrends.request import TrendReq
from functools import reduce
import time
import sys

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
    
    
    
    def _sentiment_effect(self, df):
        #print(df.columns)
        weights = df["user_followers"]*(0.5*df["user_verified"].astype(bool) + 1)
        n = weights.sum()
        mean = (df["sentiment_score"]*weights).sum()/n
        var = (weights*((df["sentiment_score"] - mean)**2)).sum()/n
        return pd.DataFrame({"sentiment_score_mean": [mean], "sentiment_score_std": [np.sqrt(var)]})
    
    def __call__(self, x):
        for func in self.preprocessing: 
            x = func(x).reset_index(drop=True)
        
        _x = x.loc[:, ["date", "user_followers", "user_verified", "sentiment_score", "period"]].groupby("period")

        #if (not os.path.exists(self.cfg.preprocessed_path)):
        #    os.mkdir(self.cfg.preprocessed_path)
        
        #for name, dataframe in _x.__iter__():
        #    dataframe.to_csv(os.path.join(self.cfg.preprocessed_path, "preprocessed_"+str(name)+".csv"))
        
        _x = _x.apply(lambda df: self._time_travel(df.groupby(pd.Grouper(key="date", freq=self._cfg.ticker_interval)).apply(self._sentiment_effect))).drop(columns=["level_1"])


        return _x

    
    def _time_travel(self, df):
        new_df = df.iloc[self._cfg.past_values:].copy()
        for i in range(1, self._cfg.past_values+1):
            #print(df.iloc[(3-i):-i].loc[:, ('sentiment_score', 'mean')].tolist()[:3])
            new_df.loc[:, ('sentiment_score_mean_n_'+str(i))] =  df.iloc[(self._cfg.past_values-i):-i].sentiment_score_mean.tolist()
        return new_df.reset_index()



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


    def __init__(self, cfg):
        #self._pytrend = TrendReq()
        self._window_length = 3
        self._cfg = cfg
        self.preprocessing = list(map(lambda f: getattr(self, f), filter(lambda method: not method.startswith('_'), dir(self))))

    def __call__(self, x):
        x = x.groupby(pd.Grouper(key="date", freq=self._cfg.ticker_interval)).mean().reset_index()
        x = self._get_returns(x)
        x = self._time_travel(self._get_log_returns(x), "log_returns")
        for f in self.preprocessing:
            x=self._time_travel(*f(x))
        return x


    def MACD_and_signal_MACD(self, x):
        exp1 = x["Close"].ewm(span=12, adjust=False).mean()
        exp2 = x["Close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        x["signal_macd"] = macd.ewm(span=9, adjust=False).mean()
        return x, "signal_macd"

    def _get_returns(self, x):
        x["returns"] = x["Close"].diff()/x["Close"] + 1
        return x

    def _get_log_returns(self, x):
        x["log_returns"] = np.log(self._get_returns(x)["returns"])
        return x


    def RSI(self, x):
        diff =  x["Close"].diff(1)
        gain =  diff.clip(lower=0).round(2)
        loss =  diff.clip(upper=0).abs().round(2)
        rs = gain.rolling(window=self._window_length, min_periods=self._window_length).mean()/loss.rolling(window=self._window_length, min_periods=self._window_length).mean()#[:self.window_length+1]
        x['rsi'] = 1.0 - (1.0 / (1.0 + rs))
        return x, "rsi"

    
    def _bollinger_bands(self, x):
        avg = x["log_returns"].rolling(window=self._window_length, min_periods=self._window_length).mean()
        std = x["log_returns"].rolling(window=self._window_length, min_periods=self._window_length).std()
        Bollinger_low, Bollinger_high = avg - 1.96*std, avg + 1.96*std
        x["distance_to_bollinger_low"], x["distance_to_bollinger_high"] = x["log_returns"] - Bollinger_low, Bollinger_high - x["log_returns"] 
        return x, "distance_to_bollinger_low", "distance_to_bollinger_high"


    def _time_travel(self, df, *names):
        #print(df.columns)
        new_df = df.iloc[self._cfg.past_values:].copy()
        for name in names:
            for i in range(1, self._cfg.past_values+1):
                #print(df.iloc[(3-i):-i].loc[:, ('sentiment_score', 'mean')].tolist()[:3])
                new_df.loc[:, name+"_n_"+str(i)] =  df.iloc[(self._cfg.past_values-i):-i].loc[:, name].tolist()
        return new_df.reset_index(drop=True)


    
    #def OBV(self, x):
    #    return np.where(x['close'] > x['close'].shift(1), x['volume'], np.where(x['close'] < x['close'].shift(1), -x['volume'], 0)).cumsum()

    def _google_trends(self, x):
        keyw = self._cfg.keyw
        start_time=self._cfg.periods[0].low
        end_time=self._cfg.periods[-1].high
        
        #year_start, month_start, day_start = self._sep_date(start_time)
        #year_end, month_end, day_end = self._sep_date(end_time)
        #day_end +=1
        names_web = {x : x+"_web" for x in keyw}
        names_news = {x : x+"_news" for x in keyw}
        
        print("loading google trends ...")
        df = self._pytrend.get_historical_interest(keyw, *self._sep_date(start_time), *self._sep_date(end_time, end=True)).rename(columns=names_web).drop(columns=["isPartial"])
        df_news = self._pytrend.get_historical_interest(keyw, *self._sep_date(start_time), *self._sep_date(end_time, end=True), gprop = 'news').rename(columns=names_news).drop(columns=["isPartial"])
        normalizing_index = pd.DataFrame(index=(pd.date_range(start=self._cfg.periods[0].low, end=self._cfg.periods[-1].high, freq="1h")))
        df = pd.merge(normalizing_index, df, left_index=True, right_index=True, how='left').reset_index().rename(columns={"index":"date"})
        df_news = pd.merge(normalizing_index, df_news, left_index=True, right_index=True, how='left').reset_index().rename(columns={"index":"date"})
        print("reattempting failed requests")
        self._reattempt_failed_requests(df, list(names_web.values()), "")
        self._reattempt_failed_requests(df_news, list(names_news.values()), "news")

        print(df.shape)

        

        df[list(names_web.values())] /=100
        df_news[list(names_news.values())] /= 100
        final = df.merge(df_news, on="date").groupby(pd.Grouper(key="date", freq=self._cfg.ticker_interval)).mean().reset_index()
        final["date"] = pd.to_datetime(final["date"], utc = True)
        print(x.columns, final.columns)
        print((final["date"] != x["date"]).values.any())
        print(final["date"])
        return x.merge(final, on="date"), *(list(names_web.values()) + list(names_news.values()))


    def _sep_date(self, date, end=False, hour=0):
        return date.year, date.month, date.day + end, date.hour*hour

    def _get_date_range_fom_mask(self, mask):
        indices = np.where(mask)[0]
        i=0
        n = len(indices)-1
        while(i < n and indices[i]+1 == indices[i+1]):
            i+=1
        return indices[0], indices[i]

    def _reattempt_failed_requests(self, df, name, type):
        _df = df.reset_index(drop=True)
        mask = _df.isna().values
        i = 1
        while (mask.any()):
            s, e = self._get_date_range_fom_mask(mask)
            start, end = _df["date"].iloc[s], _df["date"].iloc[e]
            print("fixing request timeouts for " + name[0] + " between ", start, "and", end)
            trend = self._pytrend.get_historical_interest(self._cfg.keyw, *self._sep_date(start, hour=True), *self._sep_date(end, hour=True), gprop = type)
            n = len(trend)
            if(n):
                print(len(trend))
                _df.loc[s:e, name] = trend.reset_index(drop=True)[self._cfg.keyw].values
                print(trend.iloc[:5], _df.iloc[s:s+5])
            else:
                i *= 2 
                time.sleep(60*i)
                continue
            mask = _df.isna().values
            time.sleep(60)
        return 

    

    



    #Google Search Interest,
    #Active supply Bitcoin