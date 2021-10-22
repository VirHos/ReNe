class UserProcessor:
    def __init__(self, news_dict, user_history_idx, meta_info, output_storage):
        self.news_dict = news_dict
        self.user_history_idx = user_history_idx
        self.meta_info = meta_info
        self.output_storage = output_storage

    def get_meta_info_user(self, user_id):
        history = self.user_history_idx[user_id]
        meta_str_history = [self.meta_info[idx] for idx in history]
        return meta_str_history

    def get_history(self, user_id):
        history = self.user_history_idx[user_id]
        output =  [self.output_storage[idx] for idx in history]
        return output

    def get_news_by_id(self, news_ids):
        output =  [self.output_storage[idx] for idx in news_ids]
        return output


