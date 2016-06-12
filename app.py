#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import tornado.ioloop
import tornado.web


logging.basicConfig(level=logging.INFO)


class BaseHandler(tornado.web.RequestHandler):
    '''
    '''
    # def write_error(self):
    # def get_current_user
    pass


class MainHandler(BaseHandler):

    def get(self):
        self.write("Hello, world")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    logging.info('listen on {}'.format(8888))
    tornado.ioloop.IOLoop.current().start()
