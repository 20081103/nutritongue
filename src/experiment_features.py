"""快速验证特征工程对模型性能的提升"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

DATA = r'c:\Users\云士博\AppData\Roaming\Trae CN\ModularData\ai-agent\work-mode-projects\6acca47a9de965489a99c0d3\nutritongue\data\food_database.csv'
df = pd.read_csv(DATA)

BASE = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar',
        'sodium', 'potassium', 'calcium', 'iron', 'vitamin_c']
TARGETS = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'astringent']

def build_features(frame, use_category=True):
    X = frame[BASE].astype(float).copy()
    # 对数变换（矿物质跨度大：0~38758）
    for c in ['sodium', 'potassium', 'calcium', 'iron']:
        X['log_' + c] = np.log1p(X[c])
    # 比值特征
    X['sugar_ratio'] = X['sugar'] / (X['carbs'] + 0.1)
    X['protein_density'] = X['protein'] / (X['calories'] + 1)
    X['fat_density'] = X['fat'] / (X['calories'] + 1)
    # 酸甜/咸鲜先验指标
    X['sweet_sour'] = X['sugar'] / (X['vitamin_c'] + 1)
    X['savory_idx'] = X['protein'] * np.log1p(X['sodium']) / 100
    if use_category:
        cat = pd.get_dummies(frame['category'], prefix='cat')
        X = pd.concat([X.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    return X

y = df[TARGETS].values.astype(float)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for use_cat in [False, True]:
    X = build_features(df, use_cat).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.06,
        subsample=0.9, random_state=42))
    pred = np.clip(cross_val_predict(model, Xs, y, cv=cv), 0, 10)
    r2 = r2_score(y, pred, multioutput='uniform_average')
    per = [round(r2_score(y[:, i], pred[:, i]), 3) for i in range(6)]
    print(f'category={use_cat}: CV R2={r2:.4f} per-taste={dict(zip(TARGETS, per))}')

# 随机森林对比
X = build_features(df, True).values
Xs = StandardScaler().fit_transform(X)
rf = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=300, max_depth=12, random_state=42, n_jobs=-1))
pred = np.clip(cross_val_predict(rf, Xs, y, cv=cv), 0, 10)
r2 = r2_score(y, pred, multioutput='uniform_average')
per = [round(r2_score(y[:, i], pred[:, i]), 3) for i in range(6)]
print(f'RandomForest: CV R2={r2:.4f} per-taste={dict(zip(TARGETS, per))}')
