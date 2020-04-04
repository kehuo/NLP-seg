#!/usr/bin/python
# coding=utf-8
from nlp_seg.app import app, init_app, init_app_config


def run_server(params):
    """如果 nlp.py 的 RUN_TYPE 设置为 api, 则会运行此函数
    params = nlp.py 中的 server_env 字典的值, 其中包括:
    model_path / port / host / debug 等等关键字.
    """
    init_app_config(params)
    init_app()
    debug = True if params['debug'] == 1 else False
    app.run(host=params['host'], port=params['port'], debug=debug, threaded=False)
    return
