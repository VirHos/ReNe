from typing import Dict

import numpy as np
from transformers import AutoModel, AutoTokenizer

#from tf_encoder.cache_encoder import CacheEncoder, get_nlu_executor
from encoder import CacheEncoder, Encoder, SimpleEncoder
from filters import ComplexFilter
from recommender import Recommender
from retriever import Retriever
from user_processor import UserProcessor
from utils import (build_faiss_index, get_meta_str, json_load, pickle_load,
                   yaml_load)

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

def build_user_processor(config):
    news_dict = json_load(config["news_file"])
    users = json_load(config["user_history"])

    meta_info = {}
    output_storage = {}
    url_to_index = {}

    for ix, n in enumerate(news_dict):
        output_storage[n["id"]] = {
            "id": n["id"],
            "title": n["title"],
            "data": n["date"],
        }

        meta_info[n["id"]] = get_meta_str(n)

        url_to_index[n["url"]] = n["id"]

    user_history_idx = {k: [url_to_index[i] for i in v] for k, v in users.items()}
    logger.info(f'{len(user_history_idx)} users prepared')

    user_pr = UserProcessor(
        news_dict, user_history_idx, meta_info, output_storage, url_to_index
    )
    return user_pr



def build_rene(config: Dict, stub=False):

    cache = pickle_load(config["cache_path"])
    user_pr = build_user_processor(config)

    if stub:
        logger.info('Run stub mode')
        encoder = SimpleEncoder(cache["text"], cache["embs"])
    else:
        logger.info('Loading encoder')
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
        model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
        model.eval()
        encoder = Encoder(tokenizer, model)
        encoder = CacheEncoder(encoder, cache["text"], cache["embs"])
        logger.info('Cache encoder is ready')

    output_idx_storage = list(user_pr.output_storage.keys())
    news = np.array([user_pr.meta_info[idx] for idx in output_idx_storage])
    news_embs = encoder(news)
    logger.info(f'{len(news_embs)} news prepared')
    logger.info(f'faiss index building')
    index = build_faiss_index(news_embs)
    logger.info(f'faiss index is ready')
    
    retriever = Retriever(index, encoder, np.array(output_idx_storage))

    complex_filter = ComplexFilter(user_pr)

    recomemnder = Recommender(retriever, user_pr, complex_filter)
    logger.info(f'Recommender is ready')
    return recomemnder
