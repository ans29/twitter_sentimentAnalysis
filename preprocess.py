# -*- coding: utf-8 -*-
import re
import csv
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import TweetTokenizer
from random import shuffle
from tqdm import tqdm


punctuation += '΄´’…“”–—―»«' # string.punctuation misses these.

cache_english_stopwords = stopwords.words('english')

def tweet_clean(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet_no_special_entities = re.sub(r'\&\w*;', '', tweet)
    
    # Remove tickers
    tweet_no_tickers = re.sub(r'\$\w*', '', tweet_no_special_entities)
    
    # Remove hyperlinks
    tweet_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', tweet_no_tickers)
    
    # Remove hashtags
    tweet_no_hashtags = re.sub(r'#\w*', '', tweet_no_hyperlinks)
    
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet_no_punctuation = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet_no_hashtags)
    
    # Remove words with 2 or fewer letters
    tweet_no_small_words = re.sub(r'\b\w{1,2}\b', '', tweet_no_punctuation)
    
    # Remove whitespace (including new line characters)
    tweet_no_whitespace = re.sub(r'\s\s+', ' ', tweet_no_small_words) 
    tweet_no_whitespace = tweet_no_whitespace.lstrip(' ') # Remove single space remaining at the front of the tweet.
    
	# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet_no_emojis = ''.join(c for c in tweet_no_whitespace if c <= '\uFFFF') # Apart from emojis (plane 1), this also removes historic scripts and mathematical alphanumerics (also plane 1), ideographs (plane 2) and more.
    
    # Tokenize: Change to lowercase, reduce length and remove handles
    tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) # reduce_len changes, for example, waaaaaayyyy to waaayyy.
    tw_list = tknzr.tokenize(tweet_no_emojis)
    
    # Remove stopwords
    list_no_stopwords = [i for i in tw_list if i not in cache_english_stopwords]

    # Final filtered tweet
    tweet_filtered =' '.join(list_no_stopwords)
    return tweet_filtered


# functions for cleaning
def removeStopwords(tokens):
    stops = set(stopwords.words("english"))
    stops.update(['.',',','"',"'",'?',':',';','(',')','[',']','{','}'])
    toks = [tok for tok in tokens if not tok in stops and len(tok) >= 3]
    return toks

def removeURL(text):
    newText = re.sub('http\\S+', '', text, flags=re.MULTILINE)
    return newText

def removeNum(text):
    newText = re.sub('\\d+', '', text)
    return newText

def removeHashtags(tokens):
    toks = [ tok for tok in tokens if tok[0] != '#']
    # if segment == True:
    #     segTool = Analyzer('en')
    #     for i, tag in enumerate(self.hashtags):
    #         text = tag.lstrip('#')
    #         segmented = segTool.segment(text)

    return toks

def stemTweet(tokens):
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return stemmed_words

def replace_at_the_rate(tokens):
    toks = [ tok if tok != '@' else 'at' for tok in tokens ]
    return toks

def processTweet(tweet, remove_swords = True, remove_url = True, remove_hashtags = True, remove_num = True, stem_tweet = True):
    # text = tweet.translate(string.punctuation)   -> to figure out what it does ?
    """
        Tokenize the tweet text using TweetTokenizer.
        set strip_handles = True to Twitter username handles.
        set reduce_len = True to replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    if remove_url:
        tweet = removeURL(tweet)
    if remove_num:
        tweet = removeNum(tweet)
    twtk = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = [w.lower() for w in twtk.tokenize(tweet) if w != "" and w is not None]
    if remove_hashtags:
        tokens = removeHashtags(tokens)
    if remove_swords:
        tokens = removeStopwords(tokens)
    if stem_tweet:
        tokens = stemTweet(tokens)
    tokens = replace_at_the_rate(tokens)
    text = " ".join(tokens)
    return text   



def extract_tweets(file_name):
    fw_tweets = open("tweets.txt", "w")
    with open(file_name, newline = '', encoding="ISO-8859-1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in tqdm(csv_reader):
            fw_tweets.write (str (row[0] + "%^&" + row[-1] + "\n"))


if __name__ == "__main__":
    # Preprocessing the tweets from the corpus
    # tweets = open("../Semeval2018-Task2-EmojiPrediction/train/crawler/data/tweet_by_ID_10_10_2018__01_52_23.txt.text", "r").readlines()  
    # tweet_writer = open("../Semeval2018-Task2-EmojiPrediction/train/crawler/data/processed_tweets.txt", "w")
    extract_tweets("dataset.csv")

    tweets = open("tweets.txt", "r").readlines()  
    tweet_writer = open("data.txt", "w")


    for tweet in tqdm(tweets):
        tweet = processTweet(tweet.strip(), remove_swords = True, remove_url = True, remove_hashtags = True, remove_num = True, stem_tweet = True)
        tweet_writer.write(tweet + "\n") 

    tweet_writer.close()