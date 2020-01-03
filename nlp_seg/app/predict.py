#!/usr/bin/python
# coding=utf-8
import traceback
import json
from flask_restful import reqparse, abort, Api, Resource, request
from nlp_seg.common.http import encoding_resp_utf8
from nlp_seg.app.bu_logic import predict_samples
from nlp_seg.app.init_global import global_var


class Predict(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('samples', type=dict, action='append', required=True, location='json')
        parser.add_argument('kwargs', type=dict, required=False, location='json')
        args = parser.parse_args()
        rst = predict_samples(args, global_var)
        return encoding_resp_utf8(rst)
