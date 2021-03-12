#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import numpy as np
from sklearn import linear_model, preprocessing

X = []  # get the xes
y = []  # get the ys

polynomial = preprocessing.PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)

X_ = polynomial.fit_transform(X)

model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X_, y)

model.coef_
