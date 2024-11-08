# load dependencies
import re
import nltk
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import string 

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import swifter

import tensorflow.python.keras.backend as K
from nltk.corpus import stopwords

st.set_page_config(
        page_title="Sentimen Analisis Crypto Exchange",
)

# maximum number of the allowed word in an input 
max_words = 500
# shape of input data passed for prediction
max_len = 464
# path of tokenizer file


# apply text cleaning to input data
def text_cleaning(line_from_column):
    input = line_from_column.lower()
    df = pd.DataFrame()
    # Replacing the digits/numbers
    text = text.replace('d', '')

    # ------ Tokenizing ---------

    def remove_tweet_special(text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")
                    
    text = text.apply(remove_tweet_special)

    #remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)

    text = text.apply(remove_number)

    #remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans("","",string.punctuation))

    text = text.apply(remove_punctuation)

    #remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()

    text = text.apply(remove_whitespace_LT)

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    text = text.apply(remove_whitespace_multiple)

    # remove single char
    def remove_singl_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    text = text.apply(remove_singl_char)

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    text = text.apply(word_tokenize_wrapper)

    ### Stemming ###
    # ----------------------- get stopword from NLTK stopword -------------------------------
    # get stopword indonesia
    list_stopwords = stopwords.words('indonesian')


    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                        'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                        '&amp', 'yah', 'bullbrakhma', "tokocrypto", "indodax", 'binance', 'exchange', 'crypto'])

    # ----------------------- add stopword from txt file ------------------------------------
    # read txt stopword using pandas
    #txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

    # convert stopword string to list & append additional stopword
    #list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

    # ---------------------------------------------------------------------------------------

    # convert list to dictionary
    stop = set(list_stopwords)


    text = text.apply(lambda x: [item for item in x if item not in stop])

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in text:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    print(term_dict)
    print("------------------------")


    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    text = text.swifter.apply(get_stemmed_term)

    #words = ' '.join(words)
    return text

# load the sentiment analysis model
@st.cache(allow_output_mutation=True)
def Load_model(model_pilihan):
    # path of the model
    if model_pilihan == "Tokocrypto":
        MODEL_PATH = "model_NB_tokocrypto.pkl"
        tokenizer_file = "vectorizer_NB_tokocrypto.pkl"
    if model_pilihan == "Indodax":
        MODEL_PATH = "model_NB_indodax.pkl"
        tokenizer_file = "vectorizer_NB_indodax.pkl"
    if model_pilihan == "Pintu":
        MODEL_PATH = "model_NB_pintu.pkl"
        tokenizer_file = "vectorizer_NB_pintu.pkl"

    # load tokenizer
    with open(tokenizer_file,'rb') as handle:
        tokenizer = pickle.load(handle)
    
    #coba = model_pilihan
    with open(MODEL_PATH,'rb') as handle:
        model = pickle.load(handle)
    session = K.get_session()
    return model, session, tokenizer


if __name__ == '__main__':
    st.sidebar.header("Home")
    st.title('Crypto Exchange Sentiment Analysis')
    st.write('Aplikasi sentimen analisis untuk mengetahui sentimen dari Crypto Exchange')
    st.subheader('Input tweet')
    sentence = [st.text_area('Enter your tweet here',height=200)]
    exchange_pilihan = st.radio(

        "Pilih Exchange:",

        ('Tokocrypto','Indodax','Pintu'))
    
    predict_btt = st.button('Prediksi')
    #adding a radio button

    model, session, tokenizer = Load_model(exchange_pilihan)
    

    if predict_btt:
        st.write('inilah percobaan: ')
        clean_text = []
        K.set_session(session)
        #i = text_cleaning(sentence)
        #clean_text.append(i)
        vektor = tokenizer.transform(sentence)

        # st.info(data)
        prediction = model.predict(vektor)

        st.header('Prediction using Naive Bayes model')
        if prediction == 0:
          st.error('Thread has negative sentiment')
        if prediction == 1:
          st.info('Thread has neutral sentiment')
        if prediction == 2:
          st.success('Thread has positive sentiment')