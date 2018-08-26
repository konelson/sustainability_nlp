# # sustainability_cluster_eda

from pymongo import MongoClient
client = MongoClient()
db = client.environment
sustainability_collection = db.sustainability

import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from collections import Counter

from textblob import TextBlob

import pickle

with open('sustainability_clusters_v2.pickle','rb') as read_file:
    clusters2 = pickle.load(read_file)

tweet_list = []
for tweet in sustainability_collection.aggregate([{'$match': {'lang': 'en'}}]):
    tweet_list.append(tweet['text'])


#Create Tweet DataFrame

tweet_df = pd.DataFrame()
tweet_df = pd.DataFrame(tweet_list, columns = ['tweet'])    


tweet_df['clusters'] = clusters2 #Add clusters to Dataframe


tweet_df['polarity'] = tweet_df.tweet.apply(lambda x: TextBlob(x).polarity) #Add Polarity to DataFrame
tweet_df['subjectivity'] = tweet_df.tweet.apply(lambda x: TextBlob(x).subjectivity) #Add subjectivity to Dataframe


clustered_tweets = tweet_df.filter(items = ['clusters', 'polarity', 'subjectivity'])
clustered_tweets.groupby(by = clusters).mean()

topic_counts = Counter(clusters2)

#Show percentage of Tweets per cluster:

'''
for topic, count in sorted(topic_counts.items()):
    print("Topic Number:", topic, "Percent:", round(count / sum(topic_counts.values()),2))
'''

with open ('tweet_df.pickle', 'wb') as to_write:
    pickle.dump(tweet_df, to_write)

with open ('clustered_tweets.pickle', 'wb') as to_write:
    pickle.dump(clustered_tweets, to_write)

