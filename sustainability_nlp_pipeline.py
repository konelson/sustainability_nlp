

# # sustainability_nlp_pipeline


import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import string


#Simple count vectorizer

count_vectorizer = CountVectorizer(ngram_range=(1, 2),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.5)

stop_words = stopwords.words('english')

#Add additional stop words in here:

additional_stop_words = ['RT', 'rt', "’"]

total_stop_words = stop_words + additional_stop_words

class nlp_pipeline:
    '''

    This Class outlines the NLP Pipeline being followed.

    ----------

    Inputs Include:

    vectorizer = Defaults to the Count Vectorizer, unless otherwise indicated
    tokenizer = User Input (if none, defaults to split on spaces)
    cleaning_function = User Input (if none, defaults to simple cleaning function)

    ----------

    ''' 
    def __init__(self, vectorizer = CountVectorizer(), tokenizer = None, cleaning_function = None, stemmer = None):
    
    
        if not tokenizer:
            tokenizer = self.default_tokenizer
        if not cleaning_function:
            cleaning_function = self.default_simple_cleaning
        self.tokenizer = tokenizer
        self.cleaning_function = cleaning_function
        self.stemmer = stemmer
        self.vectorizer = vectorizer
        
    def default_tokenizer(self, text):
        '''
        
        Default tokenizer that splits on spaces.
        
        '''    
        return text.split(' ')
    
    def default_simple_cleaning(self, text, tokenizer=None, stemmer=None):
        
        '''
        
        Default cleaning function that lowercases and removes punctuation.
        
        '''
    
        cleaned_text = []
        
        for post in text:
            
            post = re.sub('[%s]'% re.escape(string.punctuation), '', post).lower() #remove punctuation and lowercase
            
            cleaned_words = []
            
            for lowercase_word in tokenizer(post):    #runs tokenizer on post
                if stemmer:
                    lowercase_word = stemmer.stem(lowercase_word) #takes stem of lowercase word
                cleaned_words.append(lowercase_word)
            cleaned_text.append(' '.join(cleaned_words))
        return cleaned_text
    
    
    def fit_vectorizer(self, text):
        
        '''
        
        Runs cleaning function and then fits the vectorizer.
        
        '''
        
        clean_text = self.cleaning_function(text, self.tokenizer, self.stemmer)
        self.vectorizer.fit(clean_text)
        self._fit_check = True  #returns True if vectorizer has been fit
    
    def transform_vectorizer(self, text):
        
        '''
        
        Cleans the data and returns it in a vectorized format.
        
        '''
        
        if not self._fit_check:
            raise ValueError("Go back and fit model before transforming!")
            
        clean_text = self.cleaning_function(text, self.tokenizer, self.stemmer)
        return self.vectorizer.transform(clean_text)

    def get_features(self):

        if self.vectorizer == CountVectorizer:

            CountVectorizer.get_feature_names()


        if self.vectorizer == TfidfVectorizer:

            TfidfVectorizer.get_feature_names()


def cleaned_text(my_text, tokenizer, stemmer):
    
    tokenizer = word_tokenize
    
    words = re.sub(r"http\S+", "", my_text)                                   #remove hyperlinks
#   words = re.sub(r"@[\w_]+","",words)                                       #remove Twitter username (kept for now)
    words = re.sub('[%s]'% re.escape(string.punctuation), '', words).lower()  #remove punctuation & lowercase
    words = word_tokenize(words)                                              #Tokenize words    
#   words = regex_tokenizer.tokenize(words)                                   #Twitter specific tokenizer (turned off)

    cleaned_words = []
    
    for word in words:
    
        if word not in total_stop_words:
            word = stemmer.stem(word)
            cleaned_words.append(word)
    
    return cleaned_words

