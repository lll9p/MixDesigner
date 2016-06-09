#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
import asyncio
import os
import json
import time
from aiohttp import web
import functools


def get(path):
    '''
    Method GET decorator
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        wrapper.__method__ = 'GET'
        wrapper.__route__ = path
    return decorator


def post(path):
    '''
    Method POST decorator
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            return func(*args, **kw)
        wrapper.__method__ = 'POST'
        wrapper.__route__ = path
    return decorator

class RequestHandler():
    def __init__(self,app,func):
        self._app=app
        self._func=func
    async def __call__(self,request):
        kw = request.get?
        r = await self._func(**kw)
        return r

def add_route(app,fn):
    pass
def index(request):
    return web.Response(body=b'Hello World!')
async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', index)
    srv = await loop.create_server(app.make_handler(), '127.0.0.1', 9000)
    logging.info('server started at http://127.0.0.1:9000...')
    return srv
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init(loop))
    loop.run_forever()
