# -*- coding: utf-8 -*-
"""一次性脚本：用当前 GB 模型重新生成 281 种食品的问卷内嵌数据。

1. 替换 docs/问卷调查.html L162 的 `var FOODS=[...]`(模型预测值, 2 位小数)
2. 重导出 docs/foods_data.json(CSV 标注值, 281 条)
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from taste_model import TastePredictionModel  # noqa: E402

DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'
HTML_PATH = BASE_DIR / 'docs' / '问卷调查.html'
JSON_PATH = BASE_DIR / 'docs' / 'foods_data.json'
TASTES = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'astringent']

df = pd.read_csv(DATA_PATH)
assert len(df) == 281, f'期望281条, 实际{len(df)}'

model = TastePredictionModel()
metrics = model.train(str(DATA_PATH))
print(f"模型重训完成: R2={metrics['test_r2']:.3f} CV={metrics['cv_r2']:.3f}")

# 1) 问卷内嵌 FOODS(模型预测值)
foods_pred = []
for _, row in df.iterrows():
    nutrition = {c: float(row[c]) for c in model.feature_cols}
    pred = model.predict(nutrition)
    foods_pred.append({
        'name': row['name'],
        'category': row['category'],
        'taste': {t: pred[t] for t in TASTES},
    })
new_line = 'var FOODS=' + json.dumps(foods_pred, ensure_ascii=False) + ';'

lines = HTML_PATH.read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[161].startswith('var FOODS=['), f'L162不匹配: {lines[161][:40]}'
old_len = len(lines[161])
lines[161] = new_line + '\n'
HTML_PATH.write_text(''.join(lines), encoding='utf-8')
print(f'问卷调查.html L162 已替换: {old_len} -> {len(new_line)+1} 字符, '
      f'{len(foods_pred)} 种食品')
assert any(f['name'] == '菜籽油' for f in foods_pred), '菜籽油未包含!'

# 2) foods_data.json(CSV 标注值)
foods_label = []
for _, row in df.iterrows():
    foods_label.append({
        'name': row['name'],
        'category': row['category'],
        'taste': {t: float(row[t]) for t in TASTES},
    })
JSON_PATH.write_text(json.dumps(foods_label, ensure_ascii=False,
                                separators=(',', ':')),
                     encoding='utf-8')
print(f'foods_data.json 已重导出: {len(foods_label)} 条')

# 打印菜籽油对照
cz = [f for f in foods_pred if f['name'] == '菜籽油'][0]
print('菜籽油预测值:', json.dumps(cz, ensure_ascii=False))
