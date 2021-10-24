import sys
sys.path.append('src')

from builder import build_rene
from metrics import mapk, mapr
from utils import json_load, yaml_load


cfg = yaml_load("data/config.yml")

rene = build_rene(cfg, True)

tr = json_load("data/train_user.json")
ts = json_load("data/test_user.json")

preds = [
    [n["id"] for n in rene.get_news(uid, n_news=20)["recommendations"]]
    for uid, val in ts.items()
]
y_ts = [[rene.user_pr.url_to_index[i] for i in v] for k, v in ts.items()]

pr = mapk(y_ts, preds, 5)
rc = mapr(y_ts, preds, 20)
print(f"Recall: {rc}")
print(f"Precision: {pr}")
