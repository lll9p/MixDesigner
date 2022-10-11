#!/usr/bin/env python
# coding: utf-8
from typing import List


def revert_excel_array_attr_text(attr_text: str):
    """
    Revert defined array to python array.
    """
    return list(
        map(
            lambda line: list(
                map(lambda element: element.strip('"'), line.split(","))
            ),
            attr_text.strip("{}").split(";"),
        )
    )


def convert_excel_array_attr_text(arr: List[List[str]], number: bool = False):
    """
    Before passing arr, have to reshape to 2d list of strs.
    """
    if number:
        return "{" + ";".join(map(",".join, arr)) + "}"
    else:
        arr = [[f'"{element}"' for element in line] for line in arr]
        return "{" + ";".join(map(",".join, arr)) + "}"
