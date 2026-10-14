"""分析苦味标签分布与特征相关性，定位R²低的原因"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r'c:\Users\云士博\AppData\Roaming\Trae CN\ModularData\ai-agent\work-mode-projects\6acca47a9de965489a99c0d3\nutritongue\src')

DATA = r'c:\Users\云士博\AppData\Roaming\Trae CN\ModularData\ai-agent\work-mode-projects\6acca47a9de965489a99c0d3\nutritongue\data\food_database.csv'

df = pd.read_csv(DATA)

FEATURE_COLS = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar',
                'sodium', 'potassium', 'calcium', 'iron', 'vitamin_c',
                'organic_acid', 'polyphenol',
                'caffeine', 'alkaloid', 'tannin']
TARGET_COLS = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'astringent']

print('=== 1. 苦味标签分布 ===')
bitter = df['bitter']
print(f'样本总数: {len(df)}')
print(f'苦味均值: {bitter.mean():.3f}')
print(f'苦味中位数: {bitter.median():.3f}')
print(f'苦味标准差: {bitter.std():.3f}')
print(f'苦味最小值: {bitter.min()}')
print(f'苦味最大值: {bitter.max()}')
print()
print('苦味值分布（分桶）:')
bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 10.1]
labels = ['0','1','2','3','4','5','6','7','8','9-10']
dist = pd.cut(bitter, bins=bins, labels=labels, right=False)
print(dist.value_counts().sort_index())
print()

print('=== 2. 各味觉维度分布对比 ===')
for col in TARGET_COLS:
    print(f'{col:12s}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, '
          f'min={df[col].min()}, max={df[col].max()}, '
          f'zero_pct={((df[col]==0).sum()/len(df)*100):.1f}%')
print()

print('=== 3. 苦味与其他味觉相关性 ===')
print(df[TARGET_COLS].corr()['bitter'].sort_values(ascending=False))
print()

print('=== 4. 特征与苦味相关性 ===')
corr_bitter = df[FEATURE_COLS + ['bitter']].corr()['bitter'].drop('bitter')
print(corr_bitter.abs().sort_values(ascending=False))
print()

print('=== 5. 高苦味食品样本（bitter>=5） ===')
high_bitter = df[df['bitter'] >= 5][['name', 'category', 'bitter'] + FEATURE_COLS]
print(f'高苦味样本数: {len(high_bitter)}')
print(high_bitter[['name', 'category', 'bitter']].to_string(index=False))
print()

print('=== 6. 苦味>=3的类别分布 ===')
mid_bitter = df[df['bitter'] >= 3]
print(f'中高苦味样本数(bitter>=3): {len(mid_bitter)}')
print(mid_bitter.groupby('category').size().sort_values(ascending=False))
print()

print('=== 7. 各类别苦味均值 ===')
print(df.groupby('category')['bitter'].agg(['mean', 'max', 'count']).sort_values('mean', ascending=False))
