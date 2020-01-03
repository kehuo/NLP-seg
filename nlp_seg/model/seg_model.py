#!/usr/bin/python
# coding=utf-8

import numpy as np
import os
import re
from keras.layers import Input, Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from nlp_seg.model.crf_layer import CRF, TS, Viterbi
from nlp_seg.common.data import break_x, get_med_tag
from nlp_seg.common.date_time import fn_timer


class NLPSegModel(object):
    def __init__(self, cfg, model_path, new_model):
        self._cfg = cfg
        self._validate = True
        self._min_lr = 0.0001
        self._factor = 0.75
        self._patience = 2
        self._model_path = model_path

        if new_model:
            self._model = self._create_model()
        else:
            self._model = self._load_model()
        return

    def _create_model(self):
        cfg = self._cfg
        X = Input(shape=(cfg.seq_len, cfg.embed_dim), dtype='float32')
        Y = Input(shape=(cfg.seq_len, 1), dtype='int32')
        L = Input(shape=(1,), dtype='int32')

        # Bidirectional LSTM outputs the X encoding
        X_LSTM1 = Bidirectional(LSTM(cfg.lstm_size, return_sequences=True),
                                input_shape=(cfg.seq_len, cfg.embed_dim),
                                merge_mode='concat')(X)
        X_LSTM2 = LSTM(cfg.lstm_size, return_sequences=True)(X_LSTM1)
        XD = Dropout(rate=cfg.dropout)(X_LSTM2)

        # Dense layer maps the X encoding to tag space to get P score
        P = TimeDistributed(Dense(cfg.tag_size, activation='relu'),
                            input_shape=(cfg.seq_len, 2 * cfg.lstm_size))(XD)

        # Transition score, this is a dummy layer with only parameters
        # output tensor [tag_size, tag_size]
        A = TS(cfg.tag_size)(X)

        # CRF layer
        log_loss = CRF(cfg.batch_size)([P, A, Y, L])

        model = Model(inputs=[X, Y, L], outputs=[log_loss, P, A])

        # Only optimize log_loss, ignore P, A outputs
        rmsprop = RMSprop(lr=cfg.lr_init)
        model.compile(loss=['mse', None, None], optimizer=rmsprop)
        return model

    # Prepare model
    def _load_model(self):
        model = self._create_model()
        weights_file_path = os.path.join(self._model_path, 'lstm_weights.h5')
        model.load_weights(weights_file_path)
        return model

    # Adjust the length to the multiples of the batch_size
    def _adjust_length(self, total):
        cfg = self._cfg
        remain = total % cfg.batch_size
        return total - remain

    # get the close to 10% ratio
    def _compute_validate_ratio(self, samples):
        cfg = self._cfg
        n = samples / cfg.batch_size
        v = n / 10
        ratio = int(v) / int(n)
        return ratio

    # Compare the prediction and the golden
    def _compare_result(self, Y_pred, Y_test, L_test, verbose):
        total_bits = 0
        diff_bits = 0
        samples = Y_pred.shape[0]
        for i in range(samples):
            diff_pos = []
            for j in range(L_test[i]):
                total_bits += 1
                if Y_pred[i, j] != Y_test[i, j]:
                    diff_bits += 1
                    diff_pos.append(j)
            if verbose > 0 and len(diff_pos) > 0:
                print("L%d - %s" % (i, str(diff_pos)))
        print("Total bits: %d, diff bits: %d, accuracy: %d/1000" % (
        total_bits, diff_bits, 1000 - 1000 * diff_bits / total_bits))
        return

    # Prepare the prior knowledge of tags
    def _prepare_prior(self, yp_tags, x_len):
        cfg = self._cfg
        yp_vec = np.full(x_len, -1, dtype='int32')
        if yp_tags is not None:
            for yp_tag in yp_tags:
                (start_pos, end_pos, tag_str) = yp_tag
                tag_type = get_med_tag(tag_str, cfg.tag_dict)
                yp_vec[start_pos] = cfg.y_embedding[tag_type]
                for i in range(start_pos + 1, end_pos + 1):
                    yp_vec[i] = cfg.y_embedding['X']
        return yp_vec

    def _prepare_inputs(self, x_strs, yp_vec):
        cfg = self._cfg
        x_embedding = cfg.x_embedding
        total = 0
        samples = len(x_strs)
        X = np.zeros((samples, cfg.seq_len, cfg.embed_dim), dtype='float32')
        Y_prior = np.full((samples, cfg.seq_len), -1, dtype='int32')
        L = np.zeros(samples, dtype='int32')
        for i in range(samples):
            j = 0
            x_str = x_strs[i].split()
            L[i] = len(x_str)
            for c in x_str:
                if c in x_embedding:
                    X[i, j] = x_embedding[c]
                else:
                    X[i, j] = x_embedding["<unk>"]
                if yp_vec[total] >= 0:
                    Y_prior[i, j] = yp_vec[total]
                j += 1
                total += 1
        return (X, L, Y_prior)

    def _med_proc(self, tag_seq, offset):
        cfg = self._cfg
        y_embedding = cfg.y_embedding
        y_tag = cfg.y_tag
        # add a trailing 0
        tag_seq = np.append(tag_seq, np.zeros(1, dtype='int'))

        annos = []
        start_pos = 0
        for i in range(1, len(tag_seq)):
            if tag_seq[i] == tag_seq[i - 1] or tag_seq[i] == y_embedding['X'][0]:
                continue
            else:
                if tag_seq[i - 1] != y_embedding['Z'][0]:
                    anno = y_tag[tag_seq[start_pos]]
                    annos.append([offset + start_pos, offset + i - 1, anno])
                start_pos = i
        return annos

    def _seq_med_proc(self, tag_seq):
        cfg = self._cfg
        tags_encode_dic = cfg.tags_encode_dic
        tag_id_seq = [tags_encode_dic[tag] for tag in tag_seq]
        line_annos = self._med_proc(tag_id_seq, offset=0)
        return line_annos

    def _prepare_outputs(self, Y, L):
        cfg = self._cfg
        tags_decode_dic = cfg.tags_decode_dic
        annos = []
        offset = 0
        samples = Y.shape[0]
        tags = []
        for i in range(samples):
            row_len = L[i]
            line_annos = self._med_proc(Y[i][0:row_len], offset)
            for s in Y[i][0:row_len]:
                tags.append(tags_decode_dic[s])
            annos.extend(line_annos)
            offset += row_len
        return annos, tags

    def save_model_weights(self):
        cfg = self._cfg
        model_path = self._model_path
        model_file_path = os.path.join(model_path, 'lstm_weights.h5')
        self._model.save_weights(model_file_path)
        return

    def train_model(self, x_train, l_train, y_train, cbs=[], verbose=1):
        # Prepare training
        cfg = self._cfg
        model = self._model
        model_path = self._model_path
        model_chkpnt_path = os.path.join(model_path, 'chkpnt')

        if not os.path.exists(model_chkpnt_path):
            os.makedirs(model_chkpnt_path)

        samples = self._adjust_length(x_train.shape[0])
        loss_train = np.zeros((samples))
        validate_ratio = 0.0

        # Add default callbacks
        chkpnt_file_path = os.path.join(model_chkpnt_path, 'lstm_{epoch:03d}.h5')
        if self._validate:
            if len(cbs) == 0:
                reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                              factor=self._factor,
                                              patience=self._patience,
                                              min_lr=self._min_lr,
                                              verbose=verbose)
                cbs.append(reduce_lr)
            # Always checkpoint model
            checkpoints = ModelCheckpoint(chkpnt_file_path,
                                          monitor='val_loss',
                                          save_best_only=False,
                                          save_weights_only=True,
                                          period=5)
            cbs.append(checkpoints)
            validate_ratio = self._compute_validate_ratio(samples)
        else:
            # Always checkpoint model
            checkpoints = ModelCheckpoint(chkpnt_file_path,
                                          monitor='loss',
                                          save_best_only=False,
                                          save_weights_only=True,
                                          period=5)
            cbs.append(checkpoints)

        # Fit
        model.fit([x_train[0:samples, ],
                   y_train[0:samples, ],
                   l_train[0:samples, ]],
                  [loss_train, ],
                  validation_split=validate_ratio,
                  epochs=cfg.epochs,
                  shuffle=True,
                  batch_size=cfg.batch_size,
                  callbacks=cbs)
        return

    def test_model(self, x_test, l_test, y_test=None, Y_prior=None, verbose=0):
        # Prepare testing
        cfg = self._cfg
        model = self._model
        samples = x_test.shape[0]
        Y_pred = np.zeros((samples, cfg.seq_len), dtype='int')
        Y_dummy = np.zeros((cfg.batch_size, cfg.seq_len, 1), dtype='int')

        # Test
        A = np.zeros((cfg.tag_size+1, cfg.tag_size))
        for i in range(0, samples, cfg.batch_size):
            j = i + cfg.batch_size
            if j > samples:
                j = samples

            # Get X, L for the batch
            X_batch = np.zeros((cfg.batch_size, cfg.seq_len, cfg.embed_dim))
            L_batch = np.ones((cfg.batch_size, ))
            X_batch[0:j-i, ] = x_test[i:j, ]
            L_batch[0:j-i, ] = l_test[i:j, ]

            (loss, P, A) = model.predict_on_batch([X_batch, Y_dummy, L_batch])

            # Decode
            V = Viterbi(Y_prior)
            (Y_pred[i:j], dummy) = V.decode(P[0:j-i, ], A, l_test[i:j, ])

        # Print the A matrix
        if verbose > 0:
            print("A")
            print(A)

        # Compare the results if golden is supplied
        if y_test is not None:
            y_test = np.reshape(y_test[0:samples,], (samples, cfg.seq_len))
            self._compare_result(Y_pred, y_test, l_test, verbose)
        return Y_pred

    def _seq_med_proc(self, tag_seq):
        cfg = self._cfg
        tags_encode_dic = cfg.tags_encode_dic
        tag_id_seq = [tags_encode_dic[tag] for tag in tag_seq]
        line_annos = self._med_proc(tag_id_seq, offset=0)
        return line_annos

    @fn_timer
    def inference(self, content, entity_standard):
        cfg = self._cfg
        # med_tagger_config(batch_size)
        content_list = re.split(r'[\n|\r\n]', content)
        content = ' '.join(content_list)

        x_src = content
        yp_tags = entity_standard

        # Break long string into smaller strings to fit the model
        x_len = len(x_src)
        x_strs = break_x(x_src, cfg.seq_len)

        yp_vec = self._prepare_prior(yp_tags, x_len)

        # Prepare inputs for the model
        (X, L, Y_prior) = self._prepare_inputs(x_strs, yp_vec)

        # Do the testing
        Y = self.test_model(X, L, None, Y_prior)

        # Go back to the sequence
        annos, tag_seq = self._prepare_outputs(Y, L)

        entity_standard = self._seq_med_proc(tag_seq)
        predict = {
            'content': content,
            'entity_standard': entity_standard
        }
        return predict
