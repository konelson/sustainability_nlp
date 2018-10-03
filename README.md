# sustainability_nlp

**Sustainability exploratory data analysis using natural language processing**


1.  stream_sustainability_tweets.py
    - utilizes Twitter API to stream #sustainability tweets directly into MongoDB
2.  sustainability_nlp_pipeline.py
    - creates natural language processing pipeline including cleaning function, vectorizer & tokenizer
3.  sustainability_topic_modeling.py
    - implements NLP pipeline to test TFIDF and Count Vectorizer as well as LDA, LSA and NMF, with topics displayed as output
    - topics are clustered using mini batch kMeans and displayed in a TSNE plot
4.  sustainability_cluster_analysis.py
    - performs TextBlob sentiment anaysis on clusters
5.  word_clouds.py
    - creates word clouds for each cluster
6.  twitter_feed.py
    - generates twitter feed by gathering information on input tweet including:
          polarity & subjectivity of tweet
          tweet cluster
          10 related tweets in cluster
          polarity & subjectivity of cluster
          and word cloud of cluster
