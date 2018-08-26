
# # sustainability_topic_modeling

'''

This file creates NLP functions using the NLP Pipeline for TF-IDF and Count Vectorizer comparisons.

LDA, LSA or NMF can be selected as model inputs.

Model is currently set to run using NMF.

Outputs include:  
Topics derived in topic modeling (incl. total vocab word & frequency count)
KMeans Silhouette Score Plot
TSNE Plot

Clusters saved to new pickle file 

'''

#Load in nlp pipeline

#import sustainability_nlp_pipeline
from sustainability_nlp_pipeline import nlp_pipeline as nlp_func
from sustainability_nlp_pipeline import cleaned_text

#Load in other packages

import re
import pandas as pd
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#Connect to MongoDB

from pymongo import MongoClient
client = MongoClient()
db = client.environment
sustainability_collection = db.sustainability


stop_words = stopwords.words('english')

#Add additional stop words in here:

additional_stop_words = ['RT', 'rt', "’", "retweet", "sustainability", "sustainable", "sustainabl", "sustain"]

total_stop_words = stop_words + additional_stop_words


# **Define Parameters to use in NLP Function**

#----------TF-IDF----------#

nlp_tfidf = nlp_func(vectorizer=TfidfVectorizer(min_df=2, max_df=0.5, stop_words = total_stop_words), 
                     cleaning_function=cleaned_text, tokenizer=TreebankWordTokenizer().tokenize)#, stemmer=PorterStemmer())

english_cursor = sustainability_collection.aggregate([{'$match': {'lang': 'en'}}])#{'$sample':{'size':5}}
generate_text = (x['text'] for x in english_cursor)
nlp_tfidf.fit_vectorizer(generate_text)


english_cursor = sustainability_collection.aggregate([{'$match': {'lang': 'en'}}])#{'$sample':{'size':5}}
generate_text = (x['text'] for x in english_cursor)
tfidf_tweet = nlp_tfidf.transform_vectorizer(generate_text)

#Look at words (vector columns) using TFIDF

#nlp_tfidf.vectorizer.get_feature_names()


#----------Count Vectorizer----------#

nlp_cv = nlp_func(vectorizer=CountVectorizer(max_df=0.5, stop_words = total_stop_words), cleaning_function=cleaned_text, tokenizer=TreebankWordTokenizer().tokenize)#, stemmer=PorterStemmer())

english_cursor = sustainability_collection.aggregate([{'$match': {'lang': 'en'}}])#{'$sample':{'size':5}}
generate_text = (x['text'] for x in english_cursor)
nlp_cv.fit_vectorizer(generate_text)

english_cursor = sustainability_collection.aggregate([{'$match': {'lang': 'en'}}])#{'$sample':{'size':5}}
generate_text = (x['text'] for x in english_cursor)
cv_tweet = nlp_cv.transform_vectorizer(generate_text)

#Look at words (vector columns) using Count Vectorizer

#nlp_cv.vectorizer.get_feature_names()


#---------- Models ----------#

#Define Models

n_comp = 20

lda_cv_model = LatentDirichletAllocation(n_topics=20,
                                max_iter=10,
                                random_state=42,
                               learning_method='online')
lda_tfidf_model = LatentDirichletAllocation(n_topics=20,
                                max_iter=10,
                                random_state=42,
                               learning_method='online')
lsa_cv_model = TruncatedSVD(n_components=n_comp)
lsa_tfidf_model = TruncatedSVD(n_components=n_comp)
nmf_cv_model = NMF(n_components=n_comp)
nmf_tfidf_model = NMF(n_components=n_comp)


# Fit / Transform Models:


#lda_cv_data = lda_cv_model.fit_transform(cv_tweet)
#lda_tfidf_data = lda_tfidf_model.fit_transform(tfidf_tweet)
#lsa_cv_data = lsa_cv_model.fit_transform(cv_tweet)
#lsa_tfidf_data = lsa_tfidf_model.fit_transform(tfidf_tweet)
#nmf_cv_data = nmf_cv_model.fit_transform(cv_tweet)
nmf_tfidf_data = nmf_tfidf_model.fit_transform(tfidf_tweet)

#Save this vectorized data for later:

with open ('vectorized_tfidfdata.pickle', 'wb') as to_write:
    pickle.dump(nmf_tfidf_data, to_write)


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


#----------Display Topics for each combination of Model / Vectorizer ----------#

#LDA, Count Vectorizer:
#display_topics(lda_cv_model,nlp_cv.vectorizer.get_feature_names(),10)

#LDA, TFIDF:
#display_topics(lda_tfidf_model,nlp_tfidf.vectorizer.get_feature_names(),10)

#LSA, Count Vectorizer:
#display_topics(lsa_cv_model,nlp_cv.vectorizer.get_feature_names(),10)

#LSA, TFIDF:
#display_topics(lsa_tfidf_model,nlp_tfidf.vectorizer.get_feature_names(),10)

#NMF, Count Vectorizer:
#display_topics(nmf_cv_model,nlp_cv.vectorizer.get_feature_names(),10)

#NMF, TFIDF:
print(display_topics(nmf_tfidf_model,nlp_tfidf.vectorizer.get_feature_names(),10))

#print("Count Vectorized Words and Frequency: \n",nlp_cv.vectorizer.vocabulary_)
#print("TFIDF Words and Frequency: \n",nlp_tfidf.vectorizer.vocabulary_)

#---------- Topic Modeling graph using py LDA vis----------#

'''
import pyLDAvis, pyLDAvis.sklearn
from IPython.display import display

pyLDAvis.enable_notebook()

vis = pyLDAvis.sklearn.prepare(lda_cv_model, cv_tweet, nlp_cv.vectorizer)

display(vis)

'''

# ---------- **Cluster on Topics Using NMF ** ----------#

'''
#----------Count Vectorizer----------#

SSE = []

for k in range(2,30):
    km = MiniBatchKMeans(n_clusters = k, random_state = 42)
    km.fit(nmf_cv_data)
    labels = km.labels_
    SSE.append(km.inertia_)
    
plt.figure(dpi = 150)
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.plot(range(2,30),SSE);
#plt.ylim((0,600))

plt.savefig("cluster_plot_cv")

km = MiniBatchKMeans(n_clusters=16, random_state = 42)
nmf_clusters = km.fit_predict(nmf_cv_data)


#Silhouette Plot

visualiser = SilhouetteVisualizer(MiniBatchKMeans(n_clusters=16))
visualiser.fit(nmf_cv_data)
visualiser.poof()


# TSNE Plot

model = TSNE(n_components=2, random_state = 0, verbose = 0)
low_data = model.fit_transform(nmf_cv_data)


colors = (['crimson','b','mediumseagreen','cyan','m','y', 'k', 'orange', 'springgreen', 'deepskyblue', 'yellow', 'teal', 'navy', 'plum', 'darkslategray', 'lightcoral', 'papayawhip'])
plt.figure(dpi = 150)

for i, c, label in zip (range(16), colors, list(range(16))):
    plt.scatter(low_data[nmf_clusters == i, 0], low_data[nmf_clusters == i, 1], c=c, label = label, s = 15, alpha = 1)

plt.legend(fontsize = 10, loc = 'upper left', frameon = True, facecolor = '#FFFFFF', edgecolor = '#333333');
plt.title("Clusters with TSNE", fontsize = 12);
plt.xlim(-125,125);
plt.ylim(-125,125);
plt.ylabel("Y Axis");
plt.xlabel("X Axis");
plt.yticks(fontsize =10);
plt.xticks(fontsize = 10);

'''

#----------TFIDF----------#

#Find k

SSE = []

for k in range(2,30):
    km = MiniBatchKMeans(n_clusters = k, random_state = 42)
    km.fit(nmf_tfidf_data)
    labels = km.labels_
    SSE.append(km.inertia_)
    
plt.figure(dpi = 150)
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.plot(range(2,30),SSE);
plt.savefig("cluster_plot_tfidf")

km_tfidf = MiniBatchKMeans(n_clusters=16, random_state = 4444)
nmf_tfidf_clusters2 = km_tfidf.fit_predict(nmf_tfidf_data)

#Silhouette Plot

visualiser_tfidf = SilhouetteVisualizer(MiniBatchKMeans(n_clusters=16), random_state = 4444)
visualiser_tfidf.fit(nmf_tfidf_data)
visualiser_tfidf.poof()

#TSNE Plot

model_2 = TSNE(n_components=2, random_state = 0, verbose = 0)
low_data_2 = model_2.fit_transform(nmf_tfidf_data)

colors = (['crimson','b','mediumseagreen','cyan','m','y', 'k', 'orange', 'springgreen', 
    'deepskyblue', 'yellow', 'teal', 'navy', 'plum', 'darkslategray', 'lightcoral', 'papayawhip'])

plt.figure(dpi = 150)

for i, c, label in zip (range(16), colors, list(range(16))):
    plt.scatter(low_data_2[nmf_tfidf_clusters2 == i, 0], low_data_2[nmf_tfidf_clusters2 == i, 1], c=c, label = label, s = 15, alpha = 1)

plt.legend(fontsize = 10, loc = 'upper left', frameon = True, facecolor = '#FFFFFF', edgecolor = '#333333');
plt.title("TFIDF Clusters with TSNE", fontsize = 12);
plt.xlim(-125,125);
plt.ylim(-125,125);
plt.ylabel("Y Axis");
plt.xlabel("X Axis");
plt.yticks(fontsize =10);
plt.xticks(fontsize = 10);
plt.show()

#Save Clusters

#with open ('sustainability_clusters_v2.pickle', 'wb') as to_write:
#    pickle.dump(nmf_tfidf_clusters2, to_write)

