import streamlit as st
import time
import numpy as np
# load dependencies
import re
import nltk
import pickle
import pandas as pd
import string 

st.set_page_config(page_title="Akurasi Model")


if __name__ == '__main__':
    st.markdown("# Akurasi Model")
    st.sidebar.header("Akurasi Model")
    
    report_tokocrypto = pd.read_csv("akurasi_tokocrypto.csv")
    st.subheader('Exchange Tokocrypto')
    st.write(report_tokocrypto.head())

    report_indodax = pd.read_csv("akurasi_indodax.csv")
    st.subheader('Exchange Indodax')
    st.write(report_indodax.head())

    report_pintu = pd.read_csv("akurasi_pintu.csv")
    st.subheader('Exchange Pintu')
    st.write(report_pintu.head())