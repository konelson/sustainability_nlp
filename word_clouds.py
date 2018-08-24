from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import pandas as pd
import re, string
import pickle


from pymongo import MongoClient
client = MongoClient()
db = client.environment
sustainability_collection = db.sustainability



from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#Add additional stop words in here:
word_cloud_stop_words = ['RT', 'rt', "’", "sustainability", "sustainable", "sustainabl", "sustain", "pilot", "license", "fly", "need"]

total_stop_words = stop_words + word_cloud_stop_words



def wordcloud_text(my_text, tokenizer, stemmer):
    
    words = re.sub(r"http\S+", "", my_text)                                   #remove hyperlinks
    words = re.sub(r"@[\w_]+","",words)                                       #remove Twitter username (kept for now)
    words = re.sub('[%s]'% re.escape(string.punctuation), '', words).lower()          #remove punctuation & lowercase
    words = ''.join([i for i in words if not i.isdigit()])                    #remove numbers
    words = word_tokenize(words)                                             #Tokenize words    

    cleaned_words = []
    
    for word in words:
        
        if word.lower().replace(' ','') not in total_stop_words:
#            word = stemmer.stem(word.lower())
            cleaned_words.append(word.lower())
        
    return ' '.join(cleaned_words)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200 ,
        max_font_size=40, 
        scale=3,
        random_state=42,
        relative_scaling = 0
        
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


english_cursor = sustainability_collection.aggregate([{'$match': {'lang': 'en'}}])


tweets = []
for tweet in english_cursor:
    tweets.append(wordcloud_text(tweet['text'], tokenizer = None, stemmer = None))


words = []
for tweet in tweets:
    for word in ((word_tokenize(tweet))):
        words.append(word)


show_wordcloud(words) #show word cloud for all words

#----------Hashtags----------#

with open('hashtags.pickle','rb') as read_file:
    hashtags = pickle.load(read_file)

hashtag_list = []

for word in words:
    if word in hashtags:
        hashtag_list.append(word)


show_wordcloud(hashtag_list) #show word cloud for hashtags

#----------Clusters----------#

with open('tweet_df.pickle','rb') as read_file:
    tweet_df = pickle.load(read_file)

#tweet_df.head()

def generate_wordcloud_cluster(cluster_num):
    cluster_tweets = tweet_df[tweet_df['clusters'] == cluster_num]['tweet']
    #print(cluster_tweets)
    cluster_words = []
    for tweet in cluster_tweets:
        cluster_words.append(wordcloud_text(tweet, tokenizer = None, stemmer = None))
    return show_wordcloud(cluster_words)


input_cluster_number_here = 0

generate_wordcloud_cluster(input_cluster_number_here)

cluster_names = ['Plastic Pollution', 'Plastic Eating Enzyme', 'Daily Sustainability', 
                'Pollution Dangers', 'Green Jobs', 'Sustainable Electricity', 'Robotic Snake',
                'Smart Cities', 'AI & Sustainability', 'Drones & Sustainability', 'Solar Energy', 'Crisis in Nicaragua',
                'Climate Change', 'Sustainability in Cryptocurrency', 'Corporate Social Responsibility',
                'Ocean Pollution']

clust_names_df = pd.DataFrame({'clust_nums' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'clust_names' : cluster_names})


with open ('clust_names.pickle', 'wb') as to_write:
    pickle.dump(clust_names_df, to_write)