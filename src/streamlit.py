import streamlit as st
from utils import yaml_load
from builder import build_rene
import json
import pandas as pd

def main(config):
    st.sidebar.write("""Демо рекомендательной системы новостей для пользователей mos.ru и приложения “Моя Москва”""")
    st.sidebar.header('Меню')
    n_news = st.sidebar.number_input(label='Выберите количетво новостей', min_value= 1, max_value= 20, value= 3)

    rene = build_rene(config)

    user_id = st.number_input(label='Введите user id', min_value= 1, max_value= 267, value= 3)


    get_pred = st.button('Анализировать')
    if get_pred:
        preds_dict = rene.get_news(str(user_id), n_news)
        st.subheader('История пользователя')
        st.dataframe(pd.DataFrame(preds_dict['history']))
        st.subheader('Рекомендации для пользователя')    
        st.dataframe(pd.DataFrame(preds_dict['recommendations']))

        st.subheader('Output json')
        st.write(preds_dict)




if __name__ == '__main__':
    main(yaml_load('config.yml'))
