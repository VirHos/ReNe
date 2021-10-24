from typing import Dict, List, Any


class UserProcessor:
    def __init__(
        self,
        news_dict: Dict[str, Any],
        user_history_idx: Dict[str, Any],
        meta_info: Dict[str, Any],
        output_storage: Dict[str, Any],
        url_to_index: Dict[str, Any],
    ):
        self.news_dict = news_dict
        self.user_history_idx = user_history_idx
        self.meta_info = meta_info
        self.output_storage = output_storage
        self.url_to_index = url_to_index

    def get_meta_info_user(self, user_id):
        history = self.user_history_idx[user_id]
        meta_str_history = [self.meta_info[idx] for idx in history]
        return meta_str_history

    def get_history(self, user_id):
        history = self.user_history_idx[user_id]
        output = [self.output_storage[idx] for idx in history]
        return output

    def get_news_by_id(self, news_ids):
        output = [self.output_storage[idx] for idx in news_ids]
        return output

    def get_seen_ids(self, user_id):
        history = self.user_history_idx[user_id]
        output = [self.output_storage[idx]["id"] for idx in history]
        return output

    def get_last_date_for_user(self, user_id):
        history = self.user_history_idx[user_id]
        last_date = self.output_storage[history[-1]]["data"]
        return last_date
