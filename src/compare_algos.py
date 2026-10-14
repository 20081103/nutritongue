# -*- coding: utf-8 -*-
"""算法对比实验(281种数据库版本): 同一16特征/同一留出划分/同一增强管线"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import TastePredictionModel  # noqa: E402

m = TastePredictionModel()
df = m.load_data(str(Path(__file__).resolve().parent.parent / 'data' / 'food_database.csv'))
X = df[m.feature_cols].values.astype(float)
y = df[m.target_cols].values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['category'])
X_aug, y_aug = m._augment(X_train, y_train, rounds=3)
scaler = StandardScaler()
X_aug_s = scaler.fit_transform(X_aug)
X_test_s = scaler.transform(X_test)
bi = m.target_cols.index('bitter')
sw = np.ones(len(y_aug))
sw[y_aug[:, bi] >= 5] = 2.5
sw[y_aug[:, bi] >= 7] = 3.0
ai = m.target_cols.index('astringent')

ALGOS = {
    'Ridge': (MultiOutputRegressor(Ridge()), True),
    'RandomForest': (RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), True),
    'GradientBoosting': (MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), True),
    'MLP(64,32)': (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42), False),
}
out = {}
for name, (algo, use_sw) in ALGOS.items():
    t0 = time.perf_counter()
    if use_sw:
        algo.fit(X_aug_s, y_aug, sample_weight=sw)
    else:
        algo.fit(X_aug_s, y_aug)
    dt = time.perf_counter() - t0
    pred = np.clip(algo.predict(X_test_s), 0, 10)
    out[name] = {
        'r2': round(float(r2_score(y_test, pred, multioutput='uniform_average')), 3),
        'mae': round(float(mean_absolute_error(y_test, pred)), 3),
        'bitter_r2': round(float(r2_score(y_test[:, bi], pred[:, bi])), 3),
        'astr_r2': round(float(r2_score(y_test[:, ai], pred[:, ai])), 3),
        'train_s': round(dt, 1),
    }
    print(name, out[name], flush=True)
print('JSON:' + json.dumps(out, ensure_ascii=False))
