#!/usr/bin/env python
# coding: utf-8
from collections import OrderedDict

from .completer import Completer
from .excel import convert_excel_array_attr_text, revert_excel_array_attr_text


def print_table(table: OrderedDict, precision: int = 2, padding: int = 2):
    def is_chinese(uchar: str):
        if uchar >= "\u4e00" and uchar <= "\u9fa5":
            return True
        else:
            return False

    def len_(string: str) -> int:
        return sum(2 if is_chinese(s) else 1 for s in string)

    # Format numbers.
    table = table.copy()
    keys = list(table.keys())
    for key in table:
        table[key] = [f"{n:.{precision}f}" for n in table[key]]
    # Calculate max length.
    max_lens = OrderedDict()
    header_chinese_lens = OrderedDict()
    for key, value in table.items():
        max_len_value = max([len_(v) for v in value])
        max_len_key = len_(key)
        max_lens[key] = (
            max_len_key if max_len_key > max_len_value else max_len_value
        )
        header_chinese_lens[key] = sum(1 if is_chinese(s) else 0 for s in key)
    # Get vertical border.
    corner = "+"
    dash_y = "|"
    dash_x = "-"
    border_vertical = (
        corner
        + corner.join(dash_x * (len_ + padding) for len_ in max_lens.values())
        + corner
    )
    print(border_vertical)
    # Print header
    print(
        dash_y
        + dash_y.join(
            key.center(max_lens[key] - header_chinese_lens[key] + padding)
            for key in table.keys()
        )
        + dash_y
    )
    print(border_vertical)
    for line in zip(*list(table.values())):
        print(
            dash_y
            + dash_y.join(
                n.center(max_lens[keys[j]] + padding)
                for j, n in enumerate(line)
            )
            + dash_y
        )
    print(border_vertical)
