from nlp_seg.app.system import Ping
from nlp_seg.app.predict import Predict


def init_route(api):
    api.add_resource(Ping, '/serve/ping')
    api.add_resource(Predict, '/serve/predict_samples')
