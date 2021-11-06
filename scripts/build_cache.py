from builder import build_user_processor
from utils import json_load, pickle_dump, yaml_load
from tf_encoder.cache_encoder import get_nlu_executor

import pandas as pd

cfg = yaml_load("data/config.yml")
user_pr = build_user_processor(cfg)
meta_str_list = list(user_pr.meta_info.values())

cfg = yaml_load("data/config.yml")

nlu = get_nlu_executor(yaml_load("data/model_config.yml"))

print(f"start encoding: {len(meta_str_list)} news")
embs = nlu(meta_str_list)
fn = "data/cache.pkl"

pickle_dump({"text": meta_str_list, "embs": embs}, fn)
print(f"save cache to {fn}")
