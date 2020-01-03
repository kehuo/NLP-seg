import time
import os
import json
import codecs
import numpy as np

class NLPConfig(object):
    def __init__(self, cfg, model_path):
        self.embed_dim = cfg['EMBED_DIM']
        self.seq_len = cfg['SEQ_LEN']
        self.lstm_size = cfg['LSTM_SIZE']
        self.batch_size = cfg['BATCH_SIZE']
        self.lr_init = cfg['LR_INIT']
        self.epochs = cfg['EPOCHS']
        self.tag_dict = cfg['TAGS']
        self.dropout = cfg['DROPOUT']
        self.test_size = cfg['TEST_SIZE']
        self.build_vs_train = cfg['BUILD_VS_TRAIN']
        self.tag_size = len(self.tag_dict.keys())
        self.x_embedding = self._load_x_embedding(model_path)
        self.y_embedding, self.y_tag, self.tags_decode_dic, self.tags_encode_dic = self._init_y()
        return

    def _init_y(self):
        y_embedding = {}
        keys = sorted(self.tag_dict.keys())
        for idx, key in enumerate(keys):
            y_embedding.setdefault(self.tag_dict[key], np.asarray([idx]))

        y_tag = sorted(self.tag_dict.keys())
        tags_decode_dic = [e[0] for e in sorted([(k, idx[0]) for k, idx in y_embedding.items()], key=lambda p: p[1])]
        tags_encode_dic = dict([(k, idx[0]) for k, idx in y_embedding.items()])
        return y_embedding, y_tag, tags_decode_dic, tags_encode_dic

    # load embedding
    def _load_x_embedding(self, model_path):
        embedding_file_path = os.path.join(model_path, 'vectors.txt')
        embeddings = {}
        i = 0
        f = codecs.open(embedding_file_path, encoding='utf-8', mode='r')
        for line in f:
            values = line.split()
            if line[0] == ' ':
                word = ' '
                coefs = np.asarray(values[0:], dtype='float32')
            else:
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
            i += 1
        f.close()
        return embeddings


def load_model_config(model_cfg_path):
    with open(model_cfg_path, 'r', encoding='utf-8') as rfs:
        content = rfs.read()
        model_cfg = json.loads(content)
    return model_cfg


def init_config(model_path):
    model_cfg_path = os.path.join(model_path, 'nlp_cfg.json')
    model_cfg = load_model_config(model_cfg_path)
    cfg = NLPConfig(model_cfg, model_path)
    return cfg


def init_model(cfg, model_path, run_type):
    from nlp_seg.model.seg_model import NLPSegModel
    new_model = True if run_type == 'train' else False
    nlp_model = NLPSegModel(cfg, model_path, new_model)
    return nlp_model

