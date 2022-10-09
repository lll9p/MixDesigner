#!/usr/bin/env python
# coding: utf-8

from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, Iterable, List

import numpy as np

MixType = str | int | float | Decimal | Fraction
BoundsType = Iterable[MixType]
PointsType = np.ndarray[MixType, Any]
TestPointType = Iterable[Iterable[MixType]] | Iterable[MixType] | int
ComponentNamesType = List[str]
TargetNamesType = List[str]
DefaultsType = Dict[str, Any] | bool
