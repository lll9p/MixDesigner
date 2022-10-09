#!/usr/bin/env python
# coding: utf-8


class BaseModel:
    """Base class for all estimators in MixDesiner.
    Notes
    -----
    All estimators should reimplement methods.
    """

    def __init__(self, name: str):
        """
        Initial parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        self.name = name

    def fit(self):
        pass

    def predict(self):
        pass

    def transform(self):
        pass
