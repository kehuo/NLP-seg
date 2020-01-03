#!/usr/bin/python
# coding=utf-8
import os
import sys
from nlp_seg.api_server import run_server
from nlp_seg.train import model_train
from nlp_seg.common.utils import init_config, init_model


def init_params():
    """
    环境变量中读取
    :return:
    RUN_TYPE: train / api / inference
        train： train tagging model
        api: tagging API service
        inference: 批处理 content tagging
    MODEL_PATH: 模型文件目录
    ORIGIN_PATH: 训练文件目录 (train模式)
        存放 goldset json 文件
    TRAIN_FILE: 训练文件名称
        默认为 goldset.json
    LIMIT: n
        读取goldset记录数, 0 为全部读取
    SERVICE_NAME: 服务名称， NLPTagging
    HOST: API服务监听地址
    PORT: API服务监听端口
    DEBUG：DEBUG模式, default is False
    LOG_PATH: 服务日志目录
    BACKUP_COUNT: 日志保存周期
    """
    server_env = {
        'run_type': ['RUN_TYPE', 'api'],
        'model_path': ['MODEL_PATH', './nlp_seg/data/model'],
        'origin_path': ['ORIGIN_PATH', './nlp_seg/data/origin'],
        'train_file': ['TRAIN_FILE', 'goldset.json'],
        'limit': ['LIMIT', 500],
        'service_name': ['SERVICE_NAME', 'NLPTagging'],
        'host': ['HOST', 'localhost'],
        'port': ['PORT', 5000],
        'debug': ['DEBUG', 0],
        'log_path': ['LOG_PATH', './logs'],
        'backup_count': ['BACKUP_COUNT', 30],
        'dict_path': ['DICT_PATH', './nlp_seg/dict'],
        'debug_log': ['DEBUG_LOG', 1]
    }
    env_dict = os.environ
    params = {}
    for k, v in server_env.items():
        params.setdefault(k, env_dict.get(v[0], v[1]))
    return params


def api_server(params):
    run_server(params)
    return

def inference(params):
    model_path = params['model_path']
    run_type = params['run_type']
    cfg = init_config(model_path)
    nlp_model = init_model(cfg, model_path, run_type)
    contents = [
         "皮肤白，菲薄，头发、汗毛、睫毛、眉毛浅黄色，眼虹膜颜色浅。",
         "1个月前发现肉眼血尿，就诊外院，查尿常规尿红细胞3+，白细胞-，蛋白-。肾功正常。无尿频尿急尿痛。",
         "家属代诉患儿于6天前无明显诱因出现呕吐，伴呕吐胃内容物多次，无咖啡样物质，伴腹泻，大便稀，部分水样，每天10余次，时有哭闹，就诊我院儿科门诊，予“儿茶、亿活、维生素B6片及口服补液盐”口服，患儿症状无明显好转。5天前开始出现果酱样血便2-3次，第一次量较多，再次就诊我院儿科门诊，予补液、“头孢噻肟钠 0.45g”静脉点滴及调整胃肠道功能治疗，患儿仍有反复呕吐，大便次数逐步减少并停止排气排便，伴进行性腹胀，有间断发热，体温38℃左右，无寒战，无抽搐，无呼吸困难等不适，今儿科门诊行腹部立位片提示肠梗阻，为进一步治疗，转诊我科门诊，门诊拟急性肠套叠；急性肠梗阻收入院，患儿精神状态一般，进食水即呕吐，睡眠差，小便量减少，大便如前述，体重无明显变化。",
         "生后6个月因抬头不稳就诊外院，查生化，串联质谱，颅脑MRI等正常，后进行康复训练，1岁时出现反复癫痫发作。现3岁，讲话少，会叫人，会走路，步态不稳，有频繁微笑，常拍手，睡眠少。"
    ]

    for content in contents:
        result = nlp_model.inference(content, [])
        print(result)
    return


func_map = {
    'train': model_train,
    'api': api_server,
    'inference': inference
}


def main():
    params = init_params()
    run_type = params['run_type']

    func = func_map.get(run_type, None)
    if func is None:
        print('run_type miss match [train/evaluation/service]')
        sys.exit(-1)
    func(params)
    return


if __name__ == "__main__":
    main()
