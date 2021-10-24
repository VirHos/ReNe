import logging

from filters import ComplexFilter
from retriever import Retriever
from user_processor import UserProcessor


class Recommender:
    def __init__(
        self,
        retriever: Retriever,
        user_pr: UserProcessor,
        complex_filter: ComplexFilter,
    ):
        self.retriever = retriever
        self.user_pr = user_pr
        self.complex_filter = complex_filter
        self.logger = logging.getLogger(__name__)

    def get_history(self, user_id):
        return self.user_pr.get_history(user_id)

    def get_news(self, user_id, n_news=5, to_filter=True, top_k=32):
        user_input = self.user_pr.get_meta_info_user(user_id)
        user_history = self.user_pr.get_history(user_id)
        self.logger.debug(f"Get history with size {len(user_history)} for {user_id}")
        news_idx = self.retriever(user_input, top_k)

        output_news = self.user_pr.get_news_by_id(news_idx)
        if to_filter:
            output_news = self.complex_filter(output_news, user_id)
        return {"recommendations": output_news[:n_news], "history": user_history}

    def add_news_to_index(self, news_dict):
        pass

    def add_news_for_user(self, user_id, news_dict):
        pass
