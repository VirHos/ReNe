from builder import build_rene
from utils import json_load, pickle_dump, yaml_load

cfg = yaml_load("data/config.yml")

rene = build_rene(cfg)

meta_str_list = list(rene.user_pr.meta_info.values())

embs = rene.retriever.encoder(meta_str_list, verbose=True)

pickle_dump({"text": meta_str_list, "embs": embs}, "data/cache.pkl")
