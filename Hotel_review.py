# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 10:38:37 2022

@author: ronit
"""
#Import Required Python Libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import re
import pickle
from pickle import load
import string
import nltk
import rake_nltk
from rake_nltk import Rake
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer,word_tokenize
import streamlit as st

# Body of the application

st.sidebar.header('Input Parameters')
st.sidebar.header("![Alt Text](https://media.tenor.com/images/82505a986c4b785e7ebdfe66d6bf0097/tenor.gif)")

st.header("Hotel Review Classification")
st.title("![Alt Text](https://i.imgur.com/cIQDXf0.jpg)")
st.markdown("This application has been trained on Machine learning model - **Support Vector Machines**")

st.markdown("![Alt Text](https://i.imgur.com/BneKsiS.jpg)")
st.markdown("This application can predict if the given review is **Positive, Negative or Neutral**")

#Input your review for prediction


input_review = (st.text_area("TYPE YOUR REVIEW HERE......", """""",key = 'text'))




#Loading Both SVM and TfidfVectorizer Intelligence for deployment
svm_deploy = load(open("D:/Project/PROJECT HOTEL REVIEWS/svm_deploy.pkl", "rb"))
tf_idf_deploy = load(open("D:/Project/PROJECT HOTEL REVIEWS/tf_idf_deploy.pkl", "rb"))

#Initializing necessary functions and stop words
lemmatizer = WordNetLemmatizer()
w_tokenizer=WhitespaceTokenizer()
stop_words = pd.read_csv("D:\Project\PROJECT HOTEL REVIEWS\stop.txt",header=None,squeeze=True)

#Text cleaning Functionality
def text_clean(text):
  text=text.lower()
  text=re.sub("\d+","",text) #Remove Numbers
  text=re.sub("\[.*?\]","",text) #Remove text between square brackets
  text=re.sub("\S*https?:\S*","",text) #Remove URLs
  text=re.sub("\S*www?.\S*","",text) #Remove URLs
  text=re.sub("[%s]" % re.escape(string.punctuation),"",text) #Remove All Punctuations
  text=re.sub("\n","",text) #Remove newline space
  text=re.sub(' +', " ", text) #Remove Additional space
  text=text.split() #Split the text into list of words, i.e. tokenization
  text=[word for word in text if word not in list(stop_words)] #Remove stop words
  text=' '.join(text) #join list back to string
  return text
cleaned_text=text_clean(input_review)

#Lemmatization Functionality
def lemmatize(txt):
  list_review=[lemmatizer.lemmatize(word=word, pos=tag[0].lower()) if tag[0].lower() in ['a','r','n','v'] else word for word, tag in pos_tag(word_tokenize(txt))]
  return (' '.join([x for x in list_review if x]))

#transform text into numerical
X=tf_idf_deploy.transform([lemmatize(cleaned_text)])

# Making prediction

        
if st.button("Click to make prediction"):
    prediction = int(svm_deploy.predict(X)[0])
    
    def get_save():
        st.session_state.text = ""
    if input_review:
        with st.form(key = 'text', clear_on_submit = True):							
            button = st.form_submit_button(label='clear', on_click = get_save)
    
     
    if prediction == 0:
        st.error("This is a Negative Review!")
    elif prediction == 1:
        st.warning("This is a Neutral Review!")
    else:
        st.success("This is a Positive Review!")

# Getting Keywords
def get_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    # Below steps are for removing duplicates in the keyword lists
    unique = set()
    result = []
    for item in r.get_ranked_phrases():
        if item not in unique:
            unique.add(item)
            result.append(item)
    return result

result=get_keywords(input_review)

st.subheader("Influencing Attributes for the Review")

radio=st.sidebar.radio("Click below to get top Keywords!",("Top 10","Top 20","All"))

if radio=="Top 10":
    for word in result[:10]:
        st.markdown(word)
elif radio=="Top 20":
    for word in result[:20]:
        st.markdown(word)
else:
    for word in result:
        st.markdown(word)
