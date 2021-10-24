from builder import build_rene
from utils import json_load, pickle_dump, yaml_load
from tf_encoder.cache_encoder import get_nlu_executor

cfg = yaml_load("data/config.yml")

rene = build_rene(cfg, True)

nlu = get_nlu_executor(yaml_load("data/model_config.yml"))

tr = json_load("data/train_user.json")

meta_str_list = list(rene.user_pr.meta_info.values())

embs = nlu(meta_str_list, verbose=True)

pickle_dump({"text": meta_str_list, "embs": embs}, "data/cache.pkl")
