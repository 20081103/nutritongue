# -*- coding: utf-8 -*-
"""
算法对比与涩味优化实验
Part1: 五种回归算法在同一数据管线下的留出测试对比
Part2: 涩味长尾样本加权实验(5折交叉验证, 带样本权重)
注意: 源码字符串内一律不使用中文弯引号，避免语法错误
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import TastePredictionModel  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'

M = TastePredictionModel()
FEATS, TASTES = M.feature_cols, M.target_cols
TASTE_CN = M.TASTE_NAMES
BITTER = TASTES.index('bitter')
ASTR = TASTES.index('astringent')


def augment(X, y, rounds=3):
    X_aug, y_aug = X.copy(), y.copy()
    rng = np.random.RandomState(42)
    for _ in range(rounds):
        noise = rng.normal(0, 0.02, X.shape) * X
        X_aug = np.vstack([X_aug, X + noise])
        y_aug = np.vstack([y_aug, y])
    return X_aug, y_aug


def bitter_weights(y_aug):
    sw = np.ones(len(y_aug))
    sw[y_aug[:, BITTER] >= 5] = 2.5
    sw[y_aug[:, BITTER] >= 7] = 3.0
    return sw


def astr_weights(y_aug, w5, w7):
    sw = np.ones(len(y_aug))
    sw[y_aug[:, ASTR] >= 5] = w5
    sw[y_aug[:, ASTR] >= 7] = w7
    return sw


def make_histgbr():
    return HistGradientBoostingRegressor(
        max_iter=400, max_depth=3, learning_rate=0.05,
        min_samples_leaf=8, l2_regularization=0.5,
        early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, random_state=42)


def eval_holdout(name, fit_fn):
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    X = df[FEATS].values.astype(float)
    y = df[TASTES].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['category'])
    X_aug, y_aug = augment(X_tr, y_tr)
    scaler = StandardScaler()
    X_aug_s = scaler.fit_transform(X_aug)
    sw = bitter_weights(y_aug)
    t0 = time.perf_counter()
    model = fit_fn(X_aug_s, y_aug, sw)
    elapsed = time.perf_counter() - t0
    y_pred = np.clip(model.predict(scaler.transform(X_te)), 0, 10)
    r2 = r2_score(y_te, y_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_te, y_pred)
    r2b = r2_score(y_te[:, BITTER], y_pred[:, BITTER])
    r2a = r2_score(y_te[:, ASTR], y_pred[:, ASTR])
    print(f'{name:<28} R2={r2:.3f}  MAE={mae:.3f}  '
          f'苦={r2b:.3f}  涩={r2a:.3f}  训练{elapsed:.1f}s')
    return {'name': name, 'r2': r2, 'mae': mae, 'bitter': r2b,
            'astr': r2a, 'time': elapsed}


def part1():
    print('=' * 72)
    print('Part1 算法对比 (同一16维特征/同一留出划分/同一增强, 留出20%测试)')
    print('=' * 72)
    rows = []
    rows.append(eval_holdout(
        'Ridge(线性基线)',
        lambda X, y, sw: Ridge(alpha=1.0).fit(X, y, sample_weight=sw)))
    rows.append(eval_holdout(
        'RandomForest',
        lambda X, y, sw: RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1).fit(X, y, sample_weight=sw)))
    rows.append(eval_holdout(
        'GradientBoosting',
        lambda X, y, sw: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            random_state=42)).fit(X, y, sample_weight=sw)))
    rows.append(eval_holdout(
        'MLP(64,32)',
        lambda X, y, sw: MultiOutputRegressor(MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=3000,
            early_stopping=True, random_state=42)).fit(X, y)))
    rows.append(eval_holdout(
        'HistGBR+长尾加权(本方案)',
        lambda X, y, sw: MultiOutputRegressor(make_histgbr()).fit(
            X, y, sample_weight=sw)))
    return rows


def cv_eval(weight_fn, tag):
    """手工5折CV, 折内增强+加权, 返回OOF的各味觉R2与整体R2"""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    X = df[FEATS].values.astype(float)
    y = df[TASTES].values.astype(float)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(y)
    for tr_idx, te_idx in kf.split(X):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_aug, y_aug = augment(X_tr, y_tr)
        scaler = StandardScaler()
        X_aug_s = scaler.fit_transform(X_aug)
        sw = weight_fn(y_aug)
        model = MultiOutputRegressor(make_histgbr())
        model.fit(X_aug_s, y_aug, sample_weight=sw)
        oof[te_idx] = np.clip(model.predict(scaler.transform(X[te_idx])), 0, 10)
    overall = r2_score(y, oof, multioutput='uniform_average')
    per = {c: r2_score(y[:, i], oof[:, i]) for i, c in enumerate(TASTES)}
    print(f'{tag:<24} CV整体R2={overall:.3f}  ' +
          '  '.join(f'{TASTE_CN[c]}={per[c]:.3f}' for c in TASTES))
    return overall, per


def part2():
    print()
    print('=' * 72)
    print('Part2 涩味长尾加权实验 (5折CV带权重, 决策依据)')
    print('=' * 72)
    o0, p0 = cv_eval(bitter_weights, 'V0 仅苦味加权(现方案)')
    o1, p1 = cv_eval(lambda y: np.maximum(bitter_weights(y),
                                          astr_weights(y, 2.0, 2.5)),
                     'V1 +涩味2.0/2.5')
    o2, p2 = cv_eval(lambda y: np.maximum(bitter_weights(y),
                                          astr_weights(y, 1.5, 2.0)),
                     'V2 +涩味1.5/2.0')
    print()
    for tag, o, p in [('V1', o1, p1), ('V2', o2, p2)]:
        da = p['astringent'] - p0['astringent']
        do = o - o0
        worst = min(p[c] - p0[c] for c in TASTES)
        print(f'{tag}: 涩{da:+.3f} 整体{do:+.3f} 其他味觉最大回退{worst:+.3f}')
    return (o0, p0), (o1, p1), (o2, p2)


if __name__ == '__main__':
    rows = part1()
    part2()
