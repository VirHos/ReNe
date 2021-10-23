import json
import pandas as pd

def process_users_data():
    df = pd.read_json('data/news.jsn')
    users = pd.read_excel('dataset_news_1.xlsx')
    url_title_di = dict(zip(df.url.values, df.title.values))
    users['url'] = users.url_clean.str.replace('mos.ru', '')
    users['title'] = users.url.map(url_title_di)
    users = users.drop(columns=['url_clean'])
    users = users.dropna()
    users_dict = users.groupby('user_id')['url'].agg(list).to_dict()
    return users_dict

users_dict = process_users_data()
with open('data/user_history.json', 'w') as f:
    json.dump(users_dict, f, ensure_ascii=False, indent=4)