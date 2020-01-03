#!/usr/bin/python
# coding=utf-8
import re
import os
import codecs
import json
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(100)


class Tagen():
    def __init__(self, default, get_func, proc_func, **kwargs):
        self._y_datasets = []
        self._get_func = get_func
        self._proc_func = proc_func
        self._default = default
        self._tag_dict = kwargs['tag_dict']

    def init_tags(self, seq_len):
        self._y_tags = []
        for i in range(seq_len):
            self._y_tags.append(self._default)

    def process_tags(self, jobj):
        a_tags = self._get_func(jobj)
        for a_tag in a_tags:
            # a_tag is composed of [start_pos, end_pos, tag_str]
            start_pos = a_tag[0]
            end_pos = a_tag[1]
            if len(a_tag) == 4:
                [start_pos, end_pos, tag_str, _] = a_tag
            else:
                [start_pos, end_pos, tag_str] = a_tag
            y_tags = self._proc_func(a_tag, self._tag_dict)
            self._y_tags[start_pos:end_pos+1] = y_tags

    def flush_tags(self, start_pos, end_pos):
        tag_frag = self._y_tags[start_pos:end_pos]
        tag_str = ' '.join(tag_frag)
        self._y_datasets.append(tag_str)
        # self._fw.write("%s\n" % tag_str)

    def y_tags(self):
        return self._y_tags

    def get_y_datasets(self):
        return self._y_datasets


def get_med(obj):
    return obj["target"]


def get_med_tag(tag_str, tag_dict):
    if tag_str is None:
        return tag_dict["none"]
    return tag_dict[tag_str]


def med_proc(a_tag, tag_dict):
    if len(a_tag) == 4:
        [start_pos, end_pos, tag_str, _] = a_tag
    else:
        [start_pos, end_pos, tag_str] = a_tag
    y_tag = get_med_tag(tag_str, tag_dict)
    tag_strs = []
    for i in range(start_pos, end_pos+1):
        if i == start_pos:
            tag_strs.append(y_tag)
        else:
            tag_cont = "X"
            tag_strs.append(tag_cont)
    return tag_strs


def read_json(rfile, tagens=[], seq_len=120, limit=-1):
    i = 0
    f = codecs.open(rfile, encoding='utf-8', mode='r')
    w2v_datasets = []
    x_datasets = []

    for line in f:
        if limit != 0 and i >= limit:
            break
        i += 1
        line = line.rstrip()
        jobj = json.loads(line)

        # write out space-separated x for embedding
        x = jobj["input"]["text"]
        x = x.replace('\r\n', ' ').replace('\n', ' ')
        a_str = ' '.join(x)
        w2v_datasets.append(a_str)

        # go through each tagens and put the tags into embedings
        for tagen in tagens:
            tagen.init_tags(len(x))
            tagen.process_tags(jobj)

        # break x into model processable pieces
        x_strs = break_x(x, seq_len, tagens)
        x_datasets.extend(x_strs)
    f.close()
    return w2v_datasets, x_datasets


def gen_seg_train_data(x_train, y_train):
    x1_train, y1_train = [], []
    line_dict = []
    idx = 0
    for l1, l2 in zip(x_train, y_train):
        idx += 1
        x_line = ''
        y_line = ''
        for x, y in zip(l1.split(' '), l2.split(' ')):
            if x_line != '':
                x_line += ' '
                y_line += ' '
            x_line += x
            y_line += y
            if y == 'VS':
                if x_line not in line_dict:
                    line_dict.append(x_line)
                    x1_train.append(x_line)
                    y1_train.append(y_line)
                x_line = ''
                y_line = ''
    return x1_train, y1_train


def prepare_train_data(params, cfg):
    # seg_gen = Tagen('B', get_seg, seg_proc, 'y.seg')
    origin_path = params['origin_path']
    model_path = params['model_path']
    goldset_path = os.path.join(origin_path, params['train_file'])
    med_gen = Tagen('Z', get_med, med_proc, tag_dict=cfg.tag_dict)
    all_gens = [med_gen]
    w_data, x_data = read_json(goldset_path, all_gens, cfg.seq_len, params['limit'])
    y_data = med_gen.get_y_datasets()
    x_train, x_test, y_train, y_test \
        = train_test_split(x_data, y_data, test_size=cfg.test_size, random_state=218, shuffle=False)
    if cfg.build_vs_train is True:
        x1_train, y1_train = gen_seg_train_data(x_train, y_train)
        x_train += x1_train
        y_train += y1_train
    train_files = [('x.train', x_train), ('y.train', y_train), ('x.test', x_test), ('y.test', y_test)]
    for one in train_files:
        file_path = os.path.join(model_path, one[0])
        ofs = codecs.open(file_path, 'w', encoding='utf-8')
        for item in one[1]:
            ofs.write('{}\n'.format(item))
        ofs.close()
    return


def create_train(filename, len_v, samples, embeddings, seq_len):
    # Get embedding dimension and datatype
    an_embedding = list(embeddings.values())[0]
    dim = np.shape(an_embedding)[0]
    dtype = np.dtype(an_embedding[0])

    # Create training samples
    train = np.zeros((samples, seq_len, dim), dtype=dtype)

    i = 0
    line_no = -1
    fx = codecs.open(filename, encoding='utf-8', mode='r')
    for line in fx:
        line_no += 1
        if len_v[line_no] == 0:
            continue
        words = line.split()
        j = 0
        for c in words:
            if c in embeddings:
                cvec = embeddings[c]
            else:
                print("Error: unknown char %s at line %d" % (c, line_no + 1))
                # assert(False)
                cvec = embeddings["<unk>"]
            train[i, j] = cvec
            j += 1
        # sanity check
        if len_v[line_no] != j:
            print("Error: length mismatch at line %d" % (line_no + 1))
            print("text len: %d computed: %d" % (len_v[line_no], j))
            assert(False)
        i += 1
    fx.close()

    return train


# Return 0 if line is longer than len_limit, otherwise return the line length
def get_length(line, len_limit):
    words = line.split()
    len_line = len(words)
    if len_line > len_limit:
        return 0
    else:
        return len_line


# Get the number of lines whose line lengths are shorter than "len_limit"
def get_num_samples(filename, len_limit):
    total = 0
    len_v = []
    fx = codecs.open(filename, encoding='utf-8', mode='r')
    for line in fx:
        len_line = get_length(line, len_limit)
        len_v.append(len_line)
        if len_line > 0:
            total += 1
    fx.close()
    return (total, len_v)


def load_data(params, cfg, train=True):
    # load X and Y training data
    model_path = params['model_path']
    if train is True:
        x_file_path, y_file_path = os.path.join(model_path, 'x.train'), os.path.join(model_path, 'y.train')
    else:
        x_file_path, y_file_path = os.path.join(model_path, 'x.test'), os.path.join(model_path, 'y.test')
    (samples, len_v) = get_num_samples(x_file_path, cfg.seq_len)
    print("Sample size for %s: %d" % (x_file_path, samples))

    x_data = create_train(x_file_path, len_v, samples, cfg.x_embedding, cfg.seq_len)
    len_data = np.asarray(len_v)
    l_data = len_data[np.where(len_data > 0)]
    assert(l_data.shape[0] == samples)

    print("Y embedding %s" % cfg.y_embedding)
    y_data = create_train(y_file_path, len_v, samples, cfg.y_embedding, cfg.seq_len)
    return x_data, y_data, l_data


def break_x(x, seq_len, tagens=[]):
    if len(x) == 0:
        return []

    sep = re.finditer('[，；。]', x)
    indices = [m.start(0) for m in sep]

    # add the last pos
    last_pos = len(x) - 1
    if len(indices) == 0 or indices[-1] < last_pos:
        indices.append(last_pos)

    # write out X
    x_strs = []
    start_pos = 0
    index_len = len(indices)
    for i in range(index_len):
        curr_pos = indices[i]
        next_pos = curr_pos + 1
        if x[curr_pos] == '，' and i < index_len - 1:
            next_len = indices[i + 1] - start_pos
            if next_len < seq_len:
                continue

        x_frag = x[start_pos:next_pos]
        num_batchs = (len(x_frag) // seq_len) + (0 if len(x_frag) % seq_len == 0 else 1)
        for b in range(num_batchs):
            if b == num_batchs - 1:
                x_frag_batch = x_frag[b * seq_len:]
                x_start_pos = start_pos + b * seq_len
                x_next_pos = next_pos
            else:
                x_frag_batch = x_frag[b * seq_len: (b+1) * seq_len]
                x_start_pos = start_pos + b * seq_len
                x_next_pos = start_pos + (b + 1) * seq_len if int(len(x_frag) / seq_len) > 0 else next_pos
            x_str_batch = adjust_space(x_frag_batch)
            x_strs.append(x_str_batch)
            for tagen in tagens:
                tagen.flush_tags(x_start_pos, x_next_pos)
        start_pos = next_pos
    return x_strs


def adjust_space(x):
    x_adj = []
    for i in range(len(x)):
        if x[i] in [' ', '　']:
            x_adj.append("<space>")
        elif x[i] == '\u000B':
            x_adj.append("<space>")
        else:
            x_adj.append(x[i])
    x_str = ' '.join(x_adj)
    return x_str
