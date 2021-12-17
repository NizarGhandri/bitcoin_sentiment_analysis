# bitcoin_sentiment_analysis
Running a few Baselines and performing sentiment analysis on bitcoin.


To run the project run the following commands:
1) If you are using Google Collab simply run  ``` pip install -r requirements.txt ``` in a cell. 
2) If you are running locally or on a remote server:
    - Create a conda virtual env with the   ``` conda env create -f environment.yml --name sentiment ```
    - Activate the virtual env with  ``` conda activate sentiment ```
3) If you want to use pip virtualenv
    - Install virtualenv with pip pip install ``` pip install virtualenv ```
    - Create virtualenv virtualenv  ``` virtualenv sentiment ```
    - If you are running Unix ``` source mypython/bin/activate ``` or Windows ``` mypthon\Scripts\activate ```
    - Run ``` pip install -r requirements.txt ```

Also add the data file "Bitcoin_tweets.csv" in the data folder. You can download it from [here](https://www.kaggle.com/alexandrayuliu/bitcoin-tweets-sentiment-analysis/data)

Afterwards simply run the Baseline.ipynb.
