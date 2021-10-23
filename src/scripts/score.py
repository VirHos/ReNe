from utils import yaml_load, json_load
from builder import build_rene
from metrics import mapk, mapr


cfg = yaml_load('data/config.yml')

rene = build_rene(cfg, True)

tr = json_load('data/train_user.json')
ts = json_load('data/test_user.json')

preds = [[n['id'] for n in rene.get_news(uid, n_news=20)['recommendations']] for uid, val in ts.items()]
y_ts = [[rene.user_pr.url_to_index[i] for i in v] for k,v in ts.items()]

print(preds[0])
print(y_ts[0])

pr = mapr(preds, y_ts, 20)
rc = mapr(preds, y_ts, 20)
print(f'Recall: {rc}')
print(f'Precision: {pr}')