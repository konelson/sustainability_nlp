import pandas as pd
import requests 

# Authenticate to Twitter

try:
    from requests_oauthlib import OAuth1
except ModuleNotFoundError:
    import sys
    import os

    sys.path.append('/usr/local/lib/python3.6/site-packages')
    from requests_oauthlib import OAuth1

from twitter_credentials import credentials

oauth = OAuth1(credentials["TWITTER_CONSUMER_KEY"],
               credentials["TWITTER_CONSUMER_KEY_SECRET"],
               credentials["TWITTER_ACCESS_TOKEN"],
               credentials["TWITTER_ACCESS_TOKEN_SECRET"])

import json
from pymongo import MongoClient

client = MongoClient()
db = client.environment
redtide_collection = db.sustainability

parameters = {"q": "#sustainability", "count":100, 
             "lang":"en"}

response = requests.get("https://api.twitter.com/1.1/search/tweets.json",
                        params = parameters,
                        auth=oauth)

# Just look at the first tweet:
response.json()['statuses'][0]


#Extract info out of tweet

#Put statuses into tweets, selecting parts of first tweet to print

tweets = response.json()['statuses']

def tweet_to_string(tweet):
    s = """
        Tweet body: {text}
        Hashtags: {hashtags}
        Username: {screenname}
        Bio: {description}
        Social status: {friends} friends, {followers} followers
        Location: {location}
    """.format(text=tweet['text'], hashtags=tweet['entities']['hashtags'],
               screenname=tweet['user']['screen_name'], 
               description=tweet['user']['description'],
               friends=tweet['user']['friends_count'],
               followers=tweet['user']['followers_count'],
               location=tweet['user']['location'])
    return s

print(tweet_to_string(tweets[0]))

db.sustainability.insert_many(tweets)

print("Number of tweets = ", len(tweets))


#Request next 100 off previous request

next_page_url = "https://api.twitter.com/1.1/search/tweets.json" + response.json()['search_metadata']['next_results']

response = requests.get(next_page_url, auth=oauth)

more_tweets = response.json()['statuses']

db.sustainability.insert_many(more_tweets)




#---------- **Streaming** ----------#

import tweepy

auth = tweepy.OAuthHandler(credentials["TWITTER_CONSUMER_KEY"],
                           credentials["TWITTER_CONSUMER_KEY_SECRET"])
auth.set_access_token(credentials["TWITTER_ACCESS_TOKEN"],
                      credentials["TWITTER_ACCESS_TOKEN_SECRET"])

api=tweepy.API(auth)


from tweepy import Stream
from tweepy.streaming import StreamListener
from IPython import display
from collections import deque
import json

class MyListener(StreamListener):
    def __init__(self):
        super().__init__()
        self.list_of_tweets = deque([], maxlen=5)
        
    def on_data(self, data):
        tweet_text = json.loads(data)['text']
        self.list_of_tweets.append(tweet_text)
        self.print_list_of_tweets()
    
        try:
            client = MongoClient()
            db = client.environment
    
            # Decode the JSON from Twitter
            datajson = json.loads(data)
            
            #insert the data into the mongoDB 
            db.sustainability.insert(datajson)
        except Exception as e:
            print(e)    
        
    def on_error(self, status):
        print('An Error has occured: ' + repr(status))

    def print_list_of_tweets(self):
        display.clear_output(wait=True)
        for index, tweet_text in enumerate(self.list_of_tweets):
            m='{}. {}\n\n'.format(index, tweet_text)
            print(m)
            
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['#sustainability'])

