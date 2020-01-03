#!/usr/bin/python
# coding=utf-8

def processDunMark(content, entities):
    rst = []
    for entityOne in entities:
        # [start, end, tag]
        start = entityOne[0]
        end = entityOne[1]
        tag = entityOne[2]
        if end - start == 0 and tag == 'vector_seg':
            cnTxt = content[start:end + 1]
            if cnTxt == '、' :
                continue
        rst.append(entityOne)
    return rst


class TaggingPatcher(object):
    def __init__(self):
        return

    def run(self, content, entity_standard):
        rst = processDunMark(content, entity_standard)
        return rst
