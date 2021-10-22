import json

import numpy as np
from encoder import Encoder
from recommender import Recommender
from retriever import Retriever
from user_processor import UserProcessor
from utils import *


def build_rene(config):
    with open(config['news_file'], 'r') as f:
        news_dict = json.loads(f.read())

    with open('user_history.json', 'r') as f:
        users = json.loads(f.read())

    meta_info = []
    output_storage = []
    url_to_index = {}

    for ix, n in enumerate(news_dict):
        output_storage.append({'id': n['id'],
        'title':n['title'],
        'data': n['date']})

        meta_info.append(get_meta_str(n))

        url_to_index[n['url']] = ix

    user_history_idx = {k: [url_to_index[i] for i in v] for k,v in users.items()}

    user_pr = UserProcessor(news_dict, user_history_idx, meta_info, output_storage)

    encoder = Encoder(0,0)

    news = np.array(meta_info)
    news_embs = encoder(news)

    index = build_faiss_index(news_embs)

    retriever = Retriever(index, encoder)

    complex_filter = lambda x: x

    recomemnder = Recommender(retriever, user_pr, complex_filter)
    return recomemnder
