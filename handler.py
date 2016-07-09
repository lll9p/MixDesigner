#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import json
import tornado.web
import models
_model_list = list(models.__models__.keys())


class BaseHandler(tornado.web.RequestHandler):
    '''
    '''
    # def write_error(self):
    # def get_current_user
    pass


class BaseAPIHandler(tornado.web.RequestHandler):
    '''
    '''
    # def write_error(self):
    # def get_current_user
    pass


class APIHandler(BaseAPIHandler):

    def prepare(self):
        if self.request.headers.get('Content-Type', '').startswith('application/json'):
            self.json_args = json.loads(self.request.body)
        else:
            self.json_args = None

    def get(self, *kw, **args):
        model_name, experiment_id = args.get(
            'model_name'), args.get('experiment_id')
        logging.info('model_name is {}.'.format(model_name))
        if model_name not in _model_list:
            model_name = 'missing'
            logging.info('model_name not FOUND.')
        # -------
        try:
            experiment_id = int(experiment_id)
            point = int(self.get_argument('point'))
            lowerbound = self.get_arguments('lowerbound')
        except:
            logging.info('experiment_id name not FOUND.')
        try:
            model = models.__models__[model_name](point, lower_bounds=lowerbound)
        except:
            coded=''
            logging.info('EXCEPTION!')
        result = {'coded': coded}
        # --------
        self.set_header('Content-Type', 'application/javascript')
        self.write(json.dumps(result))

    def post(self, *kw, **args):
        pass


class MainHandler(BaseHandler):

    def get(self):
        self.render(r'index.html',
                    title='MixDesigner',
                    designers={'Simplex-centroid designer': 'SimplexCentroid'}
                    )


class SimplexCentroidHandler(BaseHandler):

    def get(self):
        self.render(r'SimplexCentroid.html',
                    title='Simplex-centroid designer', items=['None'])
