from user_processor import UserProcessor
from retriever import Retriever


class Recommender:

    def __init__(self, retriever: Retriever, user_pr: UserProcessor, complex_filter):
        self.retriever = retriever
        self.user_pr = user_pr
        self.complex_filter = complex_filter

    def get_history(self, user_id):
        return self.user_pr.get_history(user_id)

    def get_news(self, user_id, n_news):
        user_input = self.user_pr.get_meta_info_user(user_id)
        user_history = self.user_pr.get_history(user_id)
        news_idx = self.retriever(user_input, n_news)

        output_news = self.user_pr.get_news_by_id(news_idx)
        #filtered_news = self.complex_filter(news)
        return {'recommendations': output_news, 'history': user_history}

class RecModel:

    def __init__(self, recomender, user_dict):
        self.recomender = recomender
        self.user_dict = user_dict

    def __call__(self, user_id):
        history = self.user_dict[user_id]
        news = self.recomender(history)
