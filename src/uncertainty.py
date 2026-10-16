# -*- coding: utf-8 -*-
"""不确定性量化模块(交叉保形预测 / Cross-Conformal Prediction)

主模型输出六维味觉点估计, 本模块为其附加 80% 预测区间, 回答"预测值有
多可信"。方法采用保形预测(Conformal Prediction): 以主模型 5 折交叉的
折外(out-of-fold)预测残差为校准集, 取各味觉维度绝对残差的 80% 分位
数作为半宽, 区间 = 点估计 +- 半宽。

相比分位数回归, 保形预测具备分布无关的有限样本覆盖保证(校准集与新
样本可交换时, 覆盖率 >= 目标置信水平), 且区间恒以主模型点估计为中心,
不会与任何已发布的点预测数字矛盾。本模块不修改主模型任何逻辑。
注意: 源码字符串内一律不使用中文弯引号
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from taste_model import TastePredictionModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'


class TasteUncertainty:
    """味觉预测区间估计器(交叉保形预测, 目标置信度80%)"""

    CONFIDENCE = 0.8

    # 出厂校准半宽(281种数据库v最终版, 5折交叉保形, 覆盖率0.806)
    # 数据库或主模型变动时须调用 train() 重新校准
    DEFAULT_MARGINS = {'sweet': 1.038, 'sour': 0.788, 'salty': 0.823,
                       'bitter': 0.749, 'umami': 1.144, 'astringent': 1.011}

    def __init__(self, pretrained=True):
        base = TastePredictionModel()
        self.feature_cols = base.FEATURE_COLS
        self.target_cols = base.TARGET_COLS
        self.taste_bearing = base.TASTE_BEARING
        self.taste_names = base.TASTE_NAMES
        # pretrained=True 直接装载出厂校准半宽(秒级), False 则需调用 train()
        self.margins = dict(self.DEFAULT_MARGINS) if pretrained else None

    @staticmethod
    def _make_base():
        """与主模型完全同构的基础回归器"""
        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, subsample=0.8,
            random_state=42)

    @staticmethod
    def _augment(X, y, rounds=3):
        """与主模型一致的高斯噪声数据增强"""
        rng = np.random.RandomState(42)
        X_aug, y_aug = X.copy(), y.copy()
        for _ in range(rounds):
            noise = rng.normal(0, 0.02, X.shape) * X
            X_aug = np.vstack([X_aug, X + noise])
            y_aug = np.vstack([y_aug, y])
        return X_aug, y_aug

    def _fit_fold(self, X_tr, y_tr, X_te):
        """单折: 增强+标准化+苦味长尾加权(与主模型训练管道一致)"""
        Xa, ya = self._augment(X_tr, y_tr)
        scaler = StandardScaler().fit(Xa)
        bi = self.target_cols.index('bitter')
        sw = np.ones(len(ya))
        sw[ya[:, bi] >= 5] = 2.5
        sw[ya[:, bi] >= 7] = 3.0
        mdl = MultiOutputRegressor(self._make_base())
        mdl.fit(scaler.transform(Xa), ya, sample_weight=sw)
        return np.clip(mdl.predict(scaler.transform(X_te)), 0, 10)

    def train(self, csv_path=None):
        """5折交叉保形校准, 得到各味觉维度区间半宽"""
        df = pd.read_csv(csv_path or DATA_PATH, encoding='utf-8-sig')
        X = df[self.feature_cols].values.astype(float)
        y = df[self.target_cols].values.astype(float)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros_like(y)
        for tr, te in cv.split(X):
            oof[te] = self._fit_fold(X[tr], y[tr], X[te])
        resid = np.abs(y - oof)
        n = len(y)
        # 保形分位数: ceil((n+1)*q)/n, 保证有限样本覆盖率 >= 目标置信度
        q = min(1.0, np.ceil((n + 1) * self.CONFIDENCE) / n)
        self.margins = {col: round(float(np.quantile(resid[:, i], q,
                                                     method='higher')), 3)
                        for i, col in enumerate(self.target_cols)}
        inside = np.zeros_like(y, dtype=bool)
        for i, col in enumerate(self.target_cols):
            inside[:, i] = resid[:, i] <= self.margins[col] + 1e-9
        per_taste = {self.taste_names[c]: round(float(inside[:, i].mean()), 3)
                     for i, c in enumerate(self.target_cols)}
        return {
            'confidence': self.CONFIDENCE,
            'conformal_q': round(q, 4),
            'n_calib': n,
            'margins': dict(self.margins),
            'coverage': round(float(inside.mean()), 4),
            'per_taste_coverage': per_taste,
            'mean_width': round(2 * float(np.mean(list(self.margins.values()))), 3),
        }

    def interval_for(self, point_pred):
        """以主模型点估计为中心生成区间, 裁剪至[0,10]

        point_pred: TastePredictionModel.predict() 的返回字典
        """
        if self.margins is None:
            raise ValueError('区间模型未校准, 请先调用 train() 或 pretrained=True')
        return {col: (round(max(0.0, point_pred[col] - self.margins[col]), 2),
                      round(min(10.0, point_pred[col] + self.margins[col]), 2))
                for col in self.target_cols}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    uq = TasteUncertainty()
    print('交叉保形校准:', uq.train())
    demo = {'calories': 2, 'protein': 0.3, 'carbs': 0, 'fat': 0, 'fiber': 0,
            'sugar': 0, 'sodium': 2, 'potassium': 49, 'calcium': 2, 'iron': 0,
            'vitamin_c': 0, 'organic_acid': 0.3, 'polyphenol': 120,
            'caffeine': 40, 'alkaloid': 0, 'tannin': 150}
    m = TastePredictionModel()
    m.train(str(DATA_PATH))
    pt = m.predict(demo)
    iv = uq.interval_for(pt)
    for k in m.target_cols:
        print(f'{m.TASTE_NAMES[k]}: {pt[k]} +- {uq.margins[k]} '
              f'=> [{iv[k][0]}, {iv[k][1]}]')
