#!/usr/bin/python
# coding=utf-8
import traceback
import json
import logging
from flask import Flask
from flask import request
from flask_cors import CORS
from flask_restful import Api
from bb_logger.logger import Logger
from nlp_seg.app.route import init_route
from nlp_seg.app.init_global import init_global

predictor = None
app = Flask(__name__)


def init_app_config(params):
    for key in params:
        app.config[key] = params[key]
    return


def init_app():
    CORS(app, resources=r'/*')
    api = Api(app)
    log_format = '%(asctime)s %(levelname)s ' + app.config['service_name'] + ' %(message)s'
    Logger(path=app.config['log_path'],
           name=app.config['service_name'],
           audit=False,
           format=log_format,
           level=logging.INFO,
           backupCount=int(app.config['backup_count']))

    init_global(app)
    init_route(api)
    return


# @app.after_request
# def after_request(response):
#     line = 'request {} {} {} {} {}'.format(request.remote_addr,
#                                            request.method,
#                                            request.scheme,
#                                            request.full_path,
#                                            response.status)
#     Logger.service(line, 'info')
#     return response


# def load_predictor():
#     global predictor
#     if predictor is None:
#         predictor = init_predictor(app.config)
#     return predictor
#
#
# @app.route('/serve/info', methods=['GET'])
# def info():
#     print(app.config['model_path'])
#     return app.config['model_path']
#
#
# @app.route('/serve/predict_samples', methods=['POST'])
# def predict_samples():
#     try:
#         data = request.json()
#         predictor = load_predictor()
#         outputs = predictor.predict_samples(samples, **kwargs)
#     except Exception as e:
#         return json.jsonify({'error_msg': str(e), 'error_code': 1})
#     return
