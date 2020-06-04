# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk 
import string
import re
import json

from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
from openie import StanfordOpenIE

%matplotlib inline
pd.set_option('display.max_colwidth', 100)

#Load Data
def load_data():
    data = pd.read_csv('./data/appl_tweets_raw.csv')
    return data
    
#View Data
tweet_df = load_data()
tweet_df.columns = [c.replace(' ', '_') for c in tweet_df.columns]
tweet_df.head()

print('Dataset size:',tweet_df.shape)
print('Columns are:',tweet_df.columns)
tweet_df.info()


df  = pd.DataFrame(tweet_df[['Tweet_Id', 'Tweet_content']])


#print Word chart

from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator

# if = influencer 

df_if = tweet_df[tweet_df['Followers']>=1000]
df_nonif = tweet_df[tweet_df['Followers']<1000]
tweet_All = " ".join(review for review in df.Tweet_content)
tweet_fromIF = " ".join(review for review in df_if.Tweet_content)
tweet_notFromIF = " ".join(review for review in df_nonif.Tweet_content)

fig, ax = plt.subplots(3, 1, figsize  = (30,30))

#generate graph

wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_if = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_fromIF)
wordcloud_nonIF = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_notFromIF)

ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_if, interpolation='bilinear')
ax[1].set_title('Tweets from Influencer Users',fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_if, interpolation='bilinear')
ax[2].set_title('Tweets from non Influencer Users',fontsize=30)
ax[2].axis('off')


# remove puncuations
string.punctuation

# Following function removes all the punctions from the sentence. 
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
    
df['Tweet_punct'] = df['Tweet_content'].apply(lambda x: remove_punct(x))
df.head(10)


#funciton to tokentize the tweet

def tweet_token(tweet):
    with StanfordOpenIE() as client:
        text = tweet
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            # convert token from coreNLP to JSON
            triple_token = json.dumps(triple)
            #triple_token = json.loads(x)
            #print(x)
            return triple_token

df['Tweet_tokens'] = df['Tweet_content'].apply(lambda x: tweet_token(x))
df.head(10)
