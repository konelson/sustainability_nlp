from pymongo import MongoClient
client = MongoClient()
db = client.environment
sustainability_collection = db.sustainability

from word_clouds import generate_wordcloud_cluster

import pandas as pd

import pickle

with open('sustainability_clusters.pickle','rb') as read_file:
    nmf_tfidf_clusters = pickle.load(read_file)

with open('vectorized_tfidfdata.pickle','rb') as read_file:
    nmf_tfidf_data = pickle.load(read_file)

with open('tweet_df.pickle','rb') as read_file:
    tweet_df = pickle.load(read_file)

with open('clust_names.pickle','rb') as read_file:
    clust_names = pickle.load(read_file)

with open('clustered_tweets.pickle','rb') as read_file:
    clustered_tweets = pickle.load(read_file)

def get_tweets(nmf_tfidf_data, nmf_tfidf_clusters, idx):
    df = pd.DataFrame(nmf_tfidf_clusters, columns = ['cluster'])
    df = df[df['cluster'] == nmf_tfidf_clusters[idx]]
    
    return(list(df.sample(10).index)) #Returns list of 10 indices with tweets in the same cluster

recommended_tweets = get_tweets(nmf_tfidf_data, nmf_tfidf_clusters, 30)

clustered_tweets = clustered_tweets.groupby(by = 'clusters', as_index = False).mean()

def print_tweets(idx,recommended_tweets):
    
    #print information on tweets
    print('Tweet:',idx,' \n')
    print('Polarity:',round(tweet_df.loc[idx].polarity,2) )
    print('Subjectivity:',round(tweet_df.loc[idx].subjectivity,2))
    print('\n',sustainability_collection.find()[idx]['text'])
    print('\n------\n')
    
    #print information on cluster
    clust_num = tweet_df.loc[idx].clusters
    cluster_name = clust_names.loc[clust_num].clust_names
    print('Cluster Name:',clust_num, cluster_name, '\n')
    print('Cluster Polarity:', round(clustered_tweets[clustered_tweets.clusters == clust_num].polarity[clust_num],2))
    print('Cluster Subjectivity:', round(clustered_tweets[clustered_tweets.clusters == clust_num].subjectivity[clust_num],2))
    
    #generate word cloud
    generate_wordcloud_cluster(clust_num)
    
    #recommend 10 additional tweets in cluster
    for rec_idx in recommended_tweets:
        print('\n --- Result --- \n')
        print(sustainability_collection.find()[rec_idx]['text'])

def show_tweet_info(index_num):
    recommended_tweets = get_tweets(nmf_tfidf_data, nmf_tfidf_clusters, index_num)
    print_tweets(index_num, recommended_tweets)

#show_tweet_info(1)

