# -*- coding: utf-8 -*-
"""食品数据库 CSV 完整性校验：字段数(以表头为准)、空行、编号唯一、名称重复、类别分布"""
import csv
from collections import Counter
from pathlib import Path

path = Path(__file__).resolve().parent.parent / 'data' / 'food_database.csv'

with open(path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    n_fields = len(header)
    print('Header fields:', n_fields)
    problems = []
    ids, names, cats = [], [], []
    n = 0
    for i, row in enumerate(reader, 2):
        n += 1
        ids.append(row[0] if row else '')
        if not row or len(row) != n_fields or any(x.strip() == '' for x in row):
            problems.append('Line %d: %d fields -> %s' % (i, len(row), row))
        else:
            names.append(row[1])
            cats.append(row[2])

print('Data rows:', n)
print('Problems:', problems if problems else 'NONE')
print('IDs unique:', 'OK' if len(ids) == len(set(ids)) else 'DUPLICATE!')
dup_names = [x for x in set(names) if names.count(x) > 1]
print('Duplicate names:', dup_names if dup_names else 'NONE')
print('Category counts:', dict(Counter(cats)))
