#!/usr/bin/python
# encoding=utf8
import re
import pickle
import base64
import traceback
from bb_logger.logger import Logger
from nlp_seg.common.date_time import fn_timer


@fn_timer
def predict_samples(args, global_var):
    try:
        predictor = global_var['predictor']
        kwargs = args.get('kwargs', {})
        samples = args['samples']
        return_source = kwargs.get('returnSource', True)
        use_patcher = kwargs.get('usePatcher', False)
        serialize = kwargs.get('serialize', True)
        predicts = []
        for sample in samples:
            content = sample['content']
            content_list = re.split(r'[\n|\r\n]', content)
            content = ' '.join(content_list)

            entity_standard = predictor['dic_tree'].search(content)
            result = predictor['model'].inference(content, entity_standard)

            if use_patcher:
                entity_standard = predictor['patcher'].run(content, result['entity_standard'])
            else:
                entity_standard = result['entity_standard']

            predict = {
                'entity_standard': entity_standard
            }

            if return_source:
                predict['content'] = content
            predicts.append(predict)

        if serialize:
            predicts = {'o': base64.b64encode(pickle.dumps(predicts)).decode()}
    except:
        error = traceback.format_exc()
        Logger.service(error, 'error')

    return predicts
