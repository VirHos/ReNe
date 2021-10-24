from transformers import AutoModel, AutoTokenizer
from builder import build_user_processor

from encoder import CacheEncoder, Encoder, SimpleEncoder
from utils import json_load, pickle_dump, yaml_load

cfg = yaml_load("data/config.yml")

user_pr = build_user_processor(cfg)
meta_str_list = list(user_pr.meta_info.values())
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")
model.eval()
encoder = Encoder(tokenizer, model)

embs = encoder(meta_str_list)

pickle_dump({"text": meta_str_list, "embs": embs}, "data/cache.pkl")
