import streamlit as st
import os
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pycaret.regression import setup, compare_models, pull, save_model

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
    st.title('Обучаем')
    target = st.selectbox('Выберите целевую переменную', df.columns)
    if st.button('Обучить'):
        setup(df, target=target)
        setup_df = pull()
        st.info('Параметры модели')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info('Модель')
        st.dataframe(compare_df)
        st.info('Лучшая модель')
        best_model
        save_model(best_model, 'best_model')


if choice == 'Скачать модель':
    st.title('Лучшая модель')
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Скачать модель', f, 'best_model.pkl')