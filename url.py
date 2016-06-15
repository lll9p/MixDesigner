#!/usr/bin/env python
# -*- coding: utf-8 -*-
from handler import MainHandler, SimplexCentroidHandler, APIHandler
url = [
    (r'/', MainHandler),
    (r'/SimplexCentroid', SimplexCentroidHandler),
    (r'/api/v1/(?P<model_name>.*?)/(?P<experiment_id>\d+)?(?P<discard>.*)?$', APIHandler),
]
