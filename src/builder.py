import numpy as np
from tf_encoder.cache_encoder import get_nlu_executor, CacheEncoder
from encoder import SimpleEncoder
from recommender import Recommender
from retriever import Retriever
from user_processor import UserProcessor
from utils import json_load, yaml_load, pickle_load, get_meta_str, build_faiss_index
from filters import ComplexFilter


def build_rene(config, stub=False):
    news_dict = json_load(config['news_file'])

    users = json_load(config['user_history'])

    cache = pickle_load(config['cache_path'])
    
    if stub:
        encoder = SimpleEncoder(cache['text'], cache['embs'])
    else:
        encoder = get_nlu_executor(yaml_load(config['model_config']))
        encoder = CacheEncoder(encoder, cache['text'], cache['embs'])
    

    meta_info = {}
    output_storage = {}
    output_idx_storage = []
    url_to_index = {}

    for ix, n in enumerate(news_dict):
        output_storage[n['id']] = {'id': n['id'],
        'title':n['title'],
        'data': n['date']}

        output_idx_storage.append(n['id'])

        meta_info[n['id']] = get_meta_str(n)

        url_to_index[n['url']] = n['id']

    user_history_idx = {k: [url_to_index[i] for i in v] for k,v in users.items()}

    user_pr = UserProcessor(news_dict, user_history_idx, meta_info, output_storage, url_to_index)

    news = np.array([meta_info[idx] for idx in output_idx_storage])
    news_embs = encoder(news)

    index = build_faiss_index(news_embs)

    retriever = Retriever(index, encoder, np.array(output_idx_storage))

    complex_filter = ComplexFilter(user_pr,[])

    recomemnder = Recommender(retriever, user_pr, complex_filter)
    return recomemnder
