from flask import Flask, request, jsonify
from builder import build_rene
from utils import yaml_load

class ApiHandler:

    def __init__(self, config):
        self.app = Flask(__name__)
        self.rene = build_rene(config)

    def get_news(self):
        user_id = str(request.args.get('user_id', '1'))
        n_news = int(request.args.get('n_news', 5))
        to_filter = bool(request.args.get('filter', True))
        out_json = self.rene.get_news(user_id, n_news=n_news, to_filter=to_filter)
        return jsonify(out_json)

    def route(self):
        return 'demo rest api'

    def run(self):
        self.app.add_url_rule('/get_news', view_func=self.get_news, methods=['GET'])
        self.app.add_url_rule('/', view_func=self.route, methods=['GET'])
        self.app.run(host='0.0.0.0', threaded=False)



if __name__ == '__main__':
    api = ApiHandler(yaml_load('data/config.yml'))
    api.run()
