from typing import Any, Dict
import json
from flask import Flask, jsonify, request

from builder import build_rene
from utils import yaml_load, timer


class ApiHandler:
    def __init__(self, config: Dict[str, Any]):
        self.app = Flask(__name__)
        self.rene = build_rene(config)
        self.config = config

    def get_news(self):
        user_id = str(request.args.get("user_id", "1"))
        n_news = int(request.args.get("n_news", 5))
        to_filter = bool(int(request.args.get("filter", 1)))
        out_json = self.rene.get_news(user_id, n_news=n_news, to_filter=to_filter)
        return jsonify(out_json)

    def append_to_history(self):
        user_id = str(request.args.get("user_id", "1"))
        news_id = int(request.args.get("news_id", 5))
        status = self.rene.add_news_for_user(user_id, news_id)
        return jsonify({'status': status})

    def add_news_to_history(self):
        data = request.form.get('news_dict')
        news_dict = json.loads(data)
        status = self.rene.add_news_to_index(news_dict)
        return jsonify({'status': status})

    def run(self):
        self.app.add_url_rule("/get_news", view_func=self.get_news, methods=["GET"])
        self.app.add_url_rule("/add_news", view_func=self.add_news_to_history, methods=["POST"])
        self.app.add_url_rule("/append_history", view_func=self.append_to_history, methods=["POST"])
        self.app.run(host='0.0.0.0', port=self.config['port'],  threaded=False)


if __name__ == "__main__":
    api = ApiHandler(yaml_load("data/config.yml"))
    api.run()
