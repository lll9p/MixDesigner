#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import tornado.ioloop
import tornado.options
import tornado.httpserver
from tornado.options import define, options
from application import application


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    print('Quit the server with Control-C.')
    logging.info('listen on {}'.format(options.port))
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    '''
    Run with python server.py --port=8888
    '''
    logging.basicConfig(level=logging.INFO)
    define("port", default=8888, help="run on th given port", type=int)
    main()
