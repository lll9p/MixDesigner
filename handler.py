#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tornado.web


class BaseHandler(tornado.web.RequestHandler):
    '''
    '''
    # def write_error(self):
    # def get_current_user
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
