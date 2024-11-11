import streamlit as st
import time
import numpy as np
# load dependencies
import re
import nltk
import pickle
import pandas as pd
import string 

st.set_page_config(page_title="Visualisasi")


if __name__ == '__main__':
    st.markdown("# Visualisasi")
    st.sidebar.header("Visualisasi")
    
    st.subheader('Exchange Tokocrypto')
    col1, col2 = st.columns(2)
    col1.image("wc_tokocrypto.png")
    col2.image("pc_tokocrypto.png")
    #st.image("wc_tokocrypto.png")

    st.subheader('Exchange Indodax')
    col3, col4 = st.columns(2)
    col3.image("wc_indodax.png")
    col4.image("pc_indodax.png")
  
    st.subheader('Exchange Pintu')
    col5, col6 = st.columns(2)
    col5.image("wc_pintu.png")
    col6.image("pc_pintu.png")