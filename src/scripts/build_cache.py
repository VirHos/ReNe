from utils import yaml_load, json_load
from builder import build_rene

from utils import pickle_dump

cfg = yaml_load('data/config.yml')

rene = build_rene(cfg)

tr = json_load('data/train_user.json')

meta_str_list = list(rene.user_pr.meta_info.values())

embs = rene.retriever.encoder(meta_str_list, verbose=True)

pickle_dump({'text': meta_str_list, 'embs': embs}, 'data/cache.pkl')