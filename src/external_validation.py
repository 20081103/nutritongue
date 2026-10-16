# -*- coding: utf-8 -*-
"""文献锚点外部效度验证模块

模型内部指标(R2/MAE)只能证明模型复现了训练标签, 无法回答"是否符合
真实的感官科学规律"。本模块选取 30 个有明确文献/标准共识的感官锚点,
检验模型预测值是否落入文献区间:

  A类-标准参照物: 国际感官分析标准(ISO 8586)与中国国家标准(GB/T 12310
      感官分析方法)规定的五大基本味标准参照物——蔗糖(甜)、氯化钠(咸)、
      柠檬酸(酸)、咖啡因/奎宁(苦)、谷氨酸钠MSG(鲜), 及其浓度-强度关系
  B类-文献共识: 食品感官科学教材与风味化学研究中的公认结论, 如海带
      谷氨酸为鲜味发现之源(池田菊苗1908)、香菇鸟苷酸、黑茶/柿子单宁
      涩感、啤酒花苦味、菠菜草酸涩感等

通过准则: 预测值落入文献区间。同时报告 +-0.5 容差命中率与平均偏差。
局限性(已在技术方案中声明): 锚点区间取自独立于本数据库标签体系的文献
共识, 但训练标签在标注时同样参考了领域共识, 故本验证检验的是模型是否
稳定复现领域共识而非过拟合标签噪声, 属于外部效度的必要非充分证据。
注意: 源码字符串内一律不使用中文弯引号
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'

# (食品, 味觉维度, 区间下限, 区间上限, 类别, 文献依据)
ANCHORS = [
    ('白糖', 'sweet', 9.0, 10.0, 'A', '蔗糖为甜味标准参照物(ISO 8586/GB/T 12310)'),
    ('食盐', 'salty', 9.0, 10.0, 'A', 'NaCl为咸味标准参照物(ISO 8586/GB/T 12310)'),
    ('味精', 'umami', 8.0, 10.0, 'A', 'MSG为鲜味标准参照物(ISO 8586)'),
    ('黑咖啡', 'bitter', 7.0, 9.0, 'A', '咖啡因为苦味标准参照物, 黑咖啡为高苦饮品'),
    ('柠檬', 'sour', 7.0, 9.0, 'A', '柠檬酸为酸味标准参照物, 柠檬含酸约5%'),
    ('食醋', 'sour', 7.0, 9.0, 'A', '醋酸为酸味参照之一, 食醋为典型高酸调料'),
    ('蜂蜜', 'sweet', 8.0, 10.0, 'B', '蜂蜜含糖约82%, 感官科学公认高甜'),
    ('可乐', 'sweet', 7.0, 9.0, 'B', '碳酸软饮料含糖量10.6%, 公认高甜饮品'),
    ('香蕉', 'sweet', 6.0, 9.0, 'B', '成熟香蕉高糖低酸, 公认甜主导水果'),
    ('苹果', 'sweet', 5.0, 7.0, 'B', '苹果甜主导微酸, 感官教材常用示例'),
    ('红茶', 'astringent', 3.0, 6.0, 'B', '茶黄素/茶红素带来中度涩感(茶叶感官审评)'),
    ('绿茶', 'bitter', 3.0, 6.0, 'B', '儿茶素+咖啡因致轻中度苦(茶叶感官审评)'),
    ('绿茶', 'astringent', 3.0, 6.0, 'B', '儿茶素涩感为绿茶特征(茶叶感官审评)'),
    ('牛奶', 'umami', 2.0, 4.0, 'B', '牛乳含游离谷氨酸, 呈清淡鲜感(乳制品感官)'),
    ('奶酪', 'umami', 5.0, 8.0, 'B', '成熟奶酪谷氨酸富集, 鲜味文献高频案例'),
    ('奶酪', 'salty', 5.0, 8.0, 'B', '奶酪加盐工艺, 中高度咸(乳制品感官)'),
    ('干海带', 'umami', 7.0, 9.0, 'B', '海带谷氨酸为鲜味发现之源(池田菊苗1908)'),
    ('香菇', 'umami', 6.0, 9.0, 'B', '香菇鸟苷酸GMP为鲜味协同经典案例'),
    ('西红柿', 'umami', 2.0, 5.0, 'B', '番茄为蔬菜中谷氨酸最高者之一(鲜味文献)'),
    ('酱油', 'salty', 8.0, 10.0, 'B', '酱油含盐约17%, 高咸调味品共识'),
    ('酱油', 'umami', 5.0, 8.0, 'B', '酿造酱油富含氨基酸态氮, 中高鲜'),
    ('咸鸭蛋', 'salty', 8.0, 10.0, 'B', '腌制蛋高盐, 高咸共识'),
    ('榨菜', 'salty', 8.0, 10.0, 'B', '腌制蔬菜高盐, 高咸共识'),
    ('鱼露', 'salty', 8.0, 10.0, 'B', '鱼露含盐约25%, 高咸共识'),
    ('柿子', 'astringent', 3.0, 6.0, 'B', '柿单宁为涩感经典案例(风味化学教材)'),
    ('红酒', 'astringent', 2.0, 5.0, 'B', '红葡萄酒单宁涩感为品酒学基础维度'),
    ('山楂', 'sour', 4.0, 7.0, 'B', '山楂有机酸富集, 酸主导共识'),
    ('黑巧克力', 'bitter', 5.0, 8.0, 'B', '高可可固形物致苦(可可碱/多酚)'),
    ('啤酒', 'bitter', 3.0, 5.0, 'B', '啤酒花异α酸致中低度苦(酿造学)'),
    ('菠菜', 'astringent', 4.0, 7.0, 'B', '菠菜草酸涩感为食品化学经典示例'),
    ('酸奶', 'sour', 4.0, 6.0, 'B', '乳酸发酵致中度酸(乳制品感官)'),
    ('豆瓣酱', 'salty', 8.0, 10.0, 'B', '发酵豆制品高盐(调味品感官共识)'),
]

TASTE_CN = {'sweet': '甜', 'sour': '酸', 'salty': '咸',
            'bitter': '苦', 'umami': '鲜', 'astringent': '涩'}


def evaluate_anchors(model, df=None, tol=0.5):
    """逐锚点预测并判定区间命中, 返回明细与汇总指标"""
    if df is None:
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    rows = []
    hits = 0
    hits_tol = 0
    dists = []
    for food, taste, lo, hi, cls, ref in ANCHORS:
        row = df[df['name'] == food]
        if row.empty:
            raise ValueError(f'锚点食品不在数据库: {food}')
        row = row.iloc[0]
        nut = {c: float(row[c]) for c in model.feature_cols}
        pred = model.predict(nut)[taste]
        hit = lo <= pred <= hi
        dist = 0.0 if hit else min(abs(pred - lo), abs(pred - hi))
        hit_tol = dist <= tol
        hits += hit
        hits_tol += hit_tol
        dists.append(dist)
        rows.append({
            'food': food, 'taste': TASTE_CN[taste], 'lo': lo, 'hi': hi,
            'pred': pred, 'hit': hit, 'hit_tol': hit_tol,
            'dist': round(dist, 2), 'cls': cls, 'ref': ref,
        })
    n = len(ANCHORS)
    return {
        'rows': rows,
        'n_anchors': n,
        'hits': hits,
        'hit_rate': round(hits / n, 4),
        'hits_tol': hits_tol,
        'hit_rate_tol': round(hits_tol / n, 4),
        'mean_dist': round(sum(dists) / n, 3),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from taste_model import TastePredictionModel
    m = TastePredictionModel()
    m.train(str(DATA_PATH))
    rs = evaluate_anchors(m)
    print(f"锚点数={rs['n_anchors']} 命中率={rs['hit_rate']} "
          f"容差命中率={rs['hit_rate_tol']} 平均偏差={rs['mean_dist']}")
    for r in rs['rows']:
        flag = 'OK' if r['hit'] else ('~~' if r['hit_tol'] else 'XX')
        print(f"[{flag}] {r['food']} {r['taste']} "
              f"文献[{r['lo']},{r['hi']}] 预测{r['pred']}")
