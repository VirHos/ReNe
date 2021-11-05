import json

import pandas as pd
import requests
import streamlit as st

from builder import build_rene
from utils import yaml_load


def main(config):
    st.sidebar.header("PyPyPy")
    st.sidebar.write(
        """Демо рекомендательной системы новостей для пользователей mos.ru и приложения “Моя Москва”"""
    )
    st.sidebar.header("Меню")
    n_news = st.sidebar.number_input(
        label="Выберите количество рекомендуемых новостей",
        min_value=1,
        max_value=20,
        value=3,
    )
    flt = st.sidebar.radio("Фильтровать выдачу", ("Да", "Нет"))

    user_id = st.number_input(
        label="Введите user id", min_value=1, max_value=267, value=5
    )

    get_pred = st.button("Анализировать")
    if get_pred:
        fl = int("Да" == flt)
        st.write(fl)
        st.write(flt)
        js = requests.get(
            "http://rene:5000/get_news",
            params={"user_id": str(user_id), "n_news": n_news, "filter": fl},
        )
        preds_dict = json.loads(js.content)
        if preds_dict["history"]:
            st.subheader("Последняя прочитанная новость")
            st.write(preds_dict["history"][-1])

        st.subheader("output json")
        st.write(preds_dict)


if __name__ == "__main__":
    main(yaml_load("data/config.yml"))
