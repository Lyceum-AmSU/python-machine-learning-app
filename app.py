import streamlit as st
import os
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

with st.sidebar:
    st.title('Streamlit')
    choice = st.radio('Навигация', ['Загрузить данные', 'Профилирование', 'Машинное обучение', 'Скачать модель'])

if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)

if choice == 'Загрузить данные':
    st.title('Загрузите данные')
    file = st.file_uploader('Загрузите датасет здесь')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sourcedata.csv', index=None)
        st.dataframe(df)

if choice == 'Профилирование':
    st.title('Отчет')
    profile = ProfileReport(df, title="Profiling Report")
    st_profile_report(profile)

if choice == 'Машинное обучение':
    pass

if choice == 'Скачать модель':
    pass