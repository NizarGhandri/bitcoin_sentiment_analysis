import numpy as np
import re
from transformers import pipeline



class TweetPreprocessor(object): 

    
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.preprocessing = list(map(lambda f: getattr(self, f), filter(lambda method: not method.startswith('_'), dir(self))))
    
    
    
    def __call__(self, x):
        for func in self.preprocessing: 
            x = func(x)
        return x.reset_index(drop=True).groupby("period")



    def filter_in_periods(self, x):
        x["period"] = x.date.apply(self.in_period)
        return x.dropna(subset=['period'])


    def remove_bots(self, x):
        x["source"]= x["source"].apply(str).str.lower()
        x = x[~x["source"].str.contains("bot")]
        return x

    def preprocess_tweet_text(self, x):
        x["text"] = x["text"].apply(str).apply(self._process_tweet)

    def numeric_verified(x):
        x['user_verified'] = x['user_verified'].apply(lambda u: int(u =='True'))

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

    def summon_bert(self, x):
        pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        x["sentiment_score"] = x.text.apply(pipe).apply(self._get_score)
        return x

    def _get_score(self, score): 
        sc = score[0]
        return sc['score'] * (2*(sc["label"] == "POSITIVE") - 1)

        

        
    def _in_period(self, dt):
        period = list(filter(lambda p: p[1].contains(dt), enumerate(self.cfg.periods))) 
        return period[0][0] if len(period) else np.NaN



class Factors(object):

    def __call__(x):
        return x
