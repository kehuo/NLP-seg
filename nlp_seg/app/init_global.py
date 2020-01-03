#!/usr/bin/python
# encoding=utf8
import sys
import os
import traceback
from nlp_seg.common.utils import init_model, init_config
from nlp_seg.model.dictionary_tree import DictionaryTree
from nlp_seg.model.tagging_patcher import TaggingPatcher

global global_var
global_var = {}


def init_info(app):
    ok = True
    err_msg = ''
    global_var['debug_log'] = app.config.get('debug_log', 0)
    return ok, err_msg


def init_predictor(app):
    ok = True
    err_msg = ''
    try:
        config = app.config
        med_dict_file = os.path.join(config['dict_path'], 'BBMEDI.dat')
        no_conflict = False
        dic_tree = DictionaryTree(no_conflict)
        dic_tree.load(med_dict_file)

        model_path = config['model_path']
        run_type = config['run_type']
        cfg = init_config(model_path)
        model = init_model(cfg, model_path, run_type)
        patcher = TaggingPatcher()
        predictor = {
            'model': model,
            'dic_tree': dic_tree,
            'patcher': patcher
        }
        global_var['predictor'] = predictor

        # ### for test
        # rst = model.inference('咳嗽2天，发热1天，咯痰。', [])
        # print(rst)

    except Exception as e:
        traceback.print_exc()
        ok = False
        err_msg = str(e)
    return ok, err_msg


init_funcs_map = [
    init_predictor,
    init_info
]


def init_global(app):
    for func in init_funcs_map:
        ok, err_msg = func(app)
        if not ok:
            print('failed to init params: ', err_msg, func)
            sys.exit(-1)
    return
