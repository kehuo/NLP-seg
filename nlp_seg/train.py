#!/usr/bin/python
# coding=utf-8
from nlp_seg.common.data import prepare_train_data, load_data
from nlp_seg.common.utils import init_config, init_model


def model_train(params):
    model_path = params['model_path']
    run_type = params['run_type']
    cfg = init_config(model_path)
    nlp_model = init_model(cfg, model_path, run_type)
    prepare_train_data(params, cfg)
    x_train, y_train, l_train = load_data(params, cfg, train=True)
    nlp_model.train_model(x_train, l_train, y_train)
    nlp_model.save_model_weights()

    # Test train, test and full
    print("Training error is:")
    nlp_model.test_model(x_train, l_train, y_test=y_train)
    print("Test error is:")
    x_test, y_test, l_test = load_data(params, cfg, train=False)
    nlp_model.test_model(x_test, l_test, y_test=y_test)
    return
