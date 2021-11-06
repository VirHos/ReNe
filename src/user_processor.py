from typing import Any, Dict, List
from utils import get_meta_str

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
        history = self.user_history_idx.get(user_id, [])
        meta_str_history = [self.meta_info[idx] for idx in history]
        return meta_str_history

    def get_history(self, user_id):
        history = self.user_history_idx.get(user_id, [])
        output = [self.output_storage[idx] for idx in history]
        return output

    def get_news_by_id(self, news_ids):
        output = [self.output_storage[idx] for idx in news_ids]
        return output

    def get_seen_ids(self, user_id):
        history = self.user_history_idx.get(user_id, [])
        output = [self.output_storage[idx]["id"] for idx in history]
        return output

    def get_last_date_for_user(self, user_id):
        history = self.user_history_idx.get(user_id, [])
        last_date = self.output_storage[history[-1]]["data"]
        return last_date

    def append_to_user_history(self, user_id, news_id):
        status = 'ok'
        if news_id not in self.output_storage:
            status = 'unknown news id'
        else:
            if user_id in self.user_history_idx:
                self.user_history_idx[user_id].append(news_id)
            else:
                self.user_history_idx[user_id] = [news_id]
        return status
    
    def add_new_news(self, news_dict):
        self.output_storage[news_dict["id"]] = {
            "id": news_dict["id"],
            "title": news_dict["title"],
            "data": news_dict["date"],
        }

        self.meta_info[news_dict["id"]] = get_meta_str(news_dict)
        self.url_to_index[news_dict["url"]] = news_dict["id"]
        return news_dict["id"]
