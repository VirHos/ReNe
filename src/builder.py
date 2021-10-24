from typing import Dict

import numpy as np

from encoder import SimpleEncoder
from filters import ComplexFilter
from recommender import Recommender
from retriever import Retriever
from tf_encoder.cache_encoder import CacheEncoder, get_nlu_executor
from user_processor import UserProcessor
from utils import build_faiss_index, get_meta_str, json_load, pickle_load, yaml_load


def build_rene(config: Dict, stub=False):
    news_dict = json_load(config["news_file"])

    users = json_load(config["user_history"])

    cache = pickle_load(config["cache_path"])

    if stub:
        encoder = SimpleEncoder(cache["text"], cache["embs"])
    else:
        encoder = get_nlu_executor(yaml_load(config["model_config"]))
        encoder = CacheEncoder(encoder, cache["text"], cache["embs"])

    meta_info = {}
    output_storage = {}
    output_idx_storage = []
    url_to_index = {}

    for ix, n in enumerate(news_dict):
        output_storage[n["id"]] = {
            "id": n["id"],
            "title": n["title"],
            "data": n["date"],
        }

        output_idx_storage.append(n["id"])

        meta_info[n["id"]] = get_meta_str(n)

        url_to_index[n["url"]] = n["id"]

    user_history_idx = {k: [url_to_index[i] for i in v] for k, v in users.items()}

    user_pr = UserProcessor(
        news_dict, user_history_idx, meta_info, output_storage, url_to_index
    )

    news = np.array([meta_info[idx] for idx in output_idx_storage])
    news_embs = encoder(news)

    index = build_faiss_index(news_embs)

    retriever = Retriever(index, encoder, np.array(output_idx_storage))

    complex_filter = ComplexFilter(user_pr)

    recomemnder = Recommender(retriever, user_pr, complex_filter)
    return recomemnder
