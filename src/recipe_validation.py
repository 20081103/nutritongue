# -*- coding: utf-8 -*-
"""真实菜谱一致性验证模块

菜谱没有数值化感官标签, 无法参与训练, 但烹饪界对经典菜式的主导味型
有明确共识(冰糖雪梨必为甜口、咸菜白粥必为咸口)。本模块将 20 道真实
菜谱按真实用料配比折算为每100g营养成分, 用正向模型+风味交互修正层
预测六维味觉, 检验主导味觉是否与烹饪共识一致。

这是独立于训练集之外的"行为级外部效度验证": 验证对象不是单个食品的
点估计, 而是模型在真实烹饪组合下的味型判别能力。
注意: 源码字符串内一律不使用中文弯引号
"""
from pathlib import Path

import pandas as pd

from taste_model import apply_taste_interactions

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'

# (菜名, [(食材, 克数)], 期望主导味觉集合, 味型说明)
# 食材必须存在于食品数据库; 期望集合按烹饪界共识给出, 命中top1或top2即一致
RECIPES = [
    # ---- 甜口类(甜品/甜菜) ----
    ('冰糖炖梨', [('梨', 250), ('白糖', 25)], {'sweet'}, '经典甜品'),
    ('拔丝红薯', [('红薯', 200), ('白糖', 40)], {'sweet'}, '甜菜代表'),
    ('红枣银耳羹', [('红枣', 50), ('干银耳', 15), ('白糖', 20)], {'sweet'}, '甜汤'),
    ('蜜红豆沙', [('红豆', 80), ('白糖', 30)], {'sweet'}, '甜馅基底'),
    ('自调奶茶', [('红茶', 150), ('牛奶', 100), ('白糖', 15)], {'sweet'}, '甜饮'),
    ('酸奶水果捞', [('苹果', 100), ('香蕉', 100), ('酸奶', 150)], {'sweet'}, '甜品'),
    # ---- 酸口类 ----
    ('糖醋排骨', [('猪肉', 150), ('食醋', 30), ('白糖', 30)], {'sour', 'sweet'}, '糖醋味型'),
    ('酸菜鱼', [('草鱼', 200), ('酸菜', 80)], {'sour', 'umami'}, '酸菜味型'),
    ('柠檬蜂蜜饮', [('柠檬', 40), ('蜂蜜', 25)], {'sour', 'sweet'}, '酸甜饮'),
    # ---- 咸口类 ----
    ('咸菜白粥', [('白粥', 250), ('榨菜', 30)], {'salty'}, '经典咸粥'),
    ('酱油炒饭', [('米饭', 200), ('酱油', 15), ('鸡蛋', 50)], {'salty', 'umami'}, '咸鲜主食'),
    ('咸蛋瘦肉粥', [('白粥', 200), ('咸鸭蛋', 50), ('猪肉', 30)], {'salty'}, '咸粥'),
    ('火腿冬瓜汤', [('火腿', 30), ('冬瓜', 150)], {'salty', 'umami'}, '咸鲜汤'),
    ('麻婆豆腐', [('豆腐', 200), ('豆瓣酱', 25)], {'salty', 'umami'}, '咸鲜辣(辣不计入)'),
    ('凉拌黄瓜', [('黄瓜', 200), ('食盐', 3), ('食醋', 10)], {'salty', 'sour'}, '凉拌咸酸'),
    # ---- 鲜口类 ----
    ('香菇炖鸡汤', [('鸡肉', 150), ('香菇', 50)], {'umami'}, '鲜味代表汤'),
    ('紫菜蛋花汤', [('紫菜', 8), ('鸡蛋', 50)], {'umami'}, '家常鲜汤'),
    ('清蒸鲈鱼', [('鲈鱼', 150), ('酱油', 10)], {'umami', 'salty'}, '清蒸海鲜'),
    # ---- 苦口类 ----
    ('凉拌苦瓜', [('苦瓜', 150), ('食盐', 2)], {'bitter'}, '苦味代表菜'),
    ('苦瓜炒蛋', [('苦瓜', 150), ('鸡蛋', 100)], {'bitter'}, '苦味家常菜'),
]

TASTE_CN = {'sweet': '甜', 'sour': '酸', 'salty': '咸',
            'bitter': '苦', 'umami': '鲜', 'astringent': '涩'}


def mix_nutrition(items, df, feature_cols):
    """按克数加权平均得到每100g营养成分"""
    total = sum(g for _, g in items)
    mix = {c: 0.0 for c in feature_cols}
    for name, grams in items:
        row = df[df['name'] == name]
        if row.empty:
            raise ValueError(f'食材不在数据库: {name}')
        row = row.iloc[0]
        for c in feature_cols:
            mix[c] += float(row[c]) * grams / total
    return mix


def evaluate_recipes(model, df=None):
    """逐菜谱预测并判定主导味觉一致性, 返回明细与汇总指标"""
    if df is None:
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    rows = []
    top1_hits = 0
    top2_hits = 0
    for name, items, expect, style in RECIPES:
        mix = mix_nutrition(items, df, model.feature_cols)
        raw = model.predict(mix)
        cor, _ = apply_taste_interactions(raw, mix)
        ranked = sorted(cor, key=cor.get, reverse=True)
        top1, top2 = ranked[0], set(ranked[:2])
        hit1 = len(expect & {top1}) > 0
        hit2 = len(expect & top2) > 0
        top1_hits += hit1
        top2_hits += hit2
        rows.append({
            'recipe': name, 'style': style,
            'expect': '/'.join(sorted(TASTE_CN[t] for t in expect)),
            'top1': TASTE_CN[top1],
            'top2': '/'.join(TASTE_CN[t] for t in ranked[:2]),
            'hit1': hit1, 'hit2': hit2,
            'profile': {k: cor[k] for k in model.target_cols},
        })
    n = len(RECIPES)
    return {
        'rows': rows,
        'n_recipes': n,
        'top1_hits': top1_hits,
        'top2_hits': top2_hits,
        'top1_rate': round(top1_hits / n, 4),
        'top2_rate': round(top2_hits / n, 4),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from taste_model import TastePredictionModel
    m = TastePredictionModel()
    m.train(str(DATA_PATH))
    rs = evaluate_recipes(m)
    print(f"菜谱数={rs['n_recipes']} top1一致率={rs['top1_rate']} "
          f"top2一致率={rs['top2_rate']}")
    for r in rs['rows']:
        flag = 'OK' if r['hit2'] else 'XX'
        print(f"[{flag}] {r['recipe']}({r['style']}): 期望{r['expect']}, "
              f"预测top1={r['top1']}, top2={r['top2']}")
