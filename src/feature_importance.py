# -*- coding: utf-8 -*-
"""模型可解释性: 排列重要性(Permutation Importance)特征归因分析
对6个味觉维度分别计算16维输入特征的重要性(打乱该特征后R2的下降量),
验证16维特征设计与食品化学先验的一致性
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import TastePredictionModel  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / 'data' / 'food_database.csv'

TASTE_CN = {'sweet': '甜味', 'sour': '酸味', 'salty': '咸味',
            'bitter': '苦味', 'umami': '鲜味', 'astringent': '涩味'}
FEAT_CN = {
    'calories': '热量', 'protein': '蛋白质', 'carbs': '碳水化合物',
    'fat': '脂肪', 'fiber': '膳食纤维', 'sugar': '糖分', 'sodium': '钠',
    'potassium': '钾', 'calcium': '钙', 'iron': '铁', 'vitamin_c': '维生素C',
    'organic_acid': '有机酸', 'polyphenol': '多酚', 'caffeine': '咖啡因',
    'alkaloid': '苦味生物碱', 'tannin': '单宁'
}


def compute_importance(n_repeats=30):
    """训练模型并返回 (16特征 x 6味觉) 的排列重要性矩阵"""
    model = TastePredictionModel()
    model.train(str(DATA))
    df = pd.read_csv(DATA)
    X = model.scaler.transform(df[model.feature_cols].values.astype(float))
    y = df[model.target_cols].values.astype(float)
    imp = np.zeros((len(model.feature_cols), len(model.target_cols)))
    for j in range(len(model.target_cols)):
        est = model.model.estimators_[j]
        r = permutation_importance(est, X, y[:, j], n_repeats=n_repeats,
                                   random_state=42)
        imp[:, j] = np.clip(r.importances_mean, 0, None)
    return imp, list(model.feature_cols), list(model.target_cols)


if __name__ == '__main__':
    imp, feats, tastes = compute_importance()
    print('=== 各味觉 Top5 重要特征 (R2下降量) ===')
    for j, t in enumerate(tastes):
        order = np.argsort(imp[:, j])[::-1][:5]
        top = '  '.join(f'{FEAT_CN[feats[i]]} {imp[i, j]:.3f}' for i in order)
        print(f'{TASTE_CN[t]}: {top}')
