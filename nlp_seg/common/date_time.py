import time
from functools import wraps
from bb_logger.logger import Logger
from nlp_seg.app.init_global import global_var


def fn_timer(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        debug_log =  global_var['debug_log']
        info = '{:s} function took {:.3f} ms'.format(func.__name__, (t1 - t0) * 1000.0)
        if debug_log == 1:
            Logger.service(info, 'info')
        print(info)
        return ret
    return new_func