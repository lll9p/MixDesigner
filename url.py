#!/usr/bin/env python
# -*- coding: utf-8 -*-
from handler import MainHandler, SimplexCentroidHandler
url = [
    (r'/', MainHandler),
    (r'/SimplexCentroid', SimplexCentroidHandler),
]
