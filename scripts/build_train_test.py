import json


def split_history(fp="data/user_history.json", n=20):
    with open(fp, "r") as f:
        ush = json.loads(f.read())

    train = {k: v[:-n] for k, v in ush.items()}
    test = {k: v[-n:] for k, v in ush.items()}

    return train, test


tr, ts = split_history()
with open("data/train_user.json", "w") as f:
    json.dump(tr, f, ensure_ascii=False, indent=4)
with open("data/test_user.json", "w") as f:
    json.dump(ts, f, ensure_ascii=False, indent=4)
