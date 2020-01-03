#!/usr/bin/python
# coding=utf-8
from nlp_seg.app import app, init_app, init_app_config

def run_server(params):
    init_app_config(params)
    init_app()
    debug = True if params['debug'] == 1 else False
    app.run(host=params['host'], port=params['port'], debug=debug, threaded=False)
    return
