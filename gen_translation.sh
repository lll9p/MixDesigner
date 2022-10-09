#!/bin/bash
xgettext -k_ -j -o locale/zh_CN/LC_MESSAGES/zh_CN.po cli.py
msgfmt -o locale/zh_CN/LC_MESSAGES/zh_CN.mo locale/zh_CN/LC_MESSAGES/zh_CN.po
