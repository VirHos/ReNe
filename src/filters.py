import numpy as np

from user_processor import UserProcessor


class ComplexFilter:
    def __init__(self, user_pr: UserProcessor):
        self.user_pr = user_pr

    def filter_seen_articles(self, pred, user):
        seen = self.user_pr.get_seen_ids(user)
        return [pr for pr in pred if pr["id"] not in seen]

    def filter_old(self, pred, user):
        ## can change to current date
        ld = self.user_pr.get_last_date_for_user(user)
        dises = [
            pr
            for pr in pred
            if np.array(pr["data"], dtype=np.datetime64)
            > np.array(ld, dtype=np.datetime64) - np.timedelta64(7, "D")
        ]
        wodises = [pr for pr in pred if pr not in dises]
        return dises + wodises

    def __call__(self, pred, user):
        filtered = self.filter_seen_articles(pred, user)
        return filtered
