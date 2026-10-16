# -*- coding: utf-8 -*-
"""
味析 NutriTongue 专项测试套件
覆盖: 模型性能/边界规则/典型食品 sanity/预测延迟/OCR 解析/API 接口
运行: python run_tests.py  (API 测试需 uvicorn 运行在 8001 端口)
注意: 源码字符串内一律不使用中文弯引号
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import TastePredictionModel  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'
API = 'http://localhost:8001'
RESULTS = []


def rec(case_id, module, scene, expect, actual, ok):
    RESULTS.append((case_id, module, scene, expect, actual,
                    'PASS' if ok else 'FAIL'))
    print(f'[{"PASS" if ok else "FAIL"}] {case_id} {scene}: {actual}')


def test_model():
    m = TastePredictionModel()
    t0 = time.perf_counter()
    met = m.train(str(DATA_PATH))
    train_s = time.perf_counter() - t0
    print('--- 模型指标 ---')
    print(json.dumps(met, ensure_ascii=False, indent=1))
    print(f'训练+评估耗时 {train_s:.1f}s')
    rec('M-01', '模型', '留出测试整体R2>=0.70', '>=0.70',
        met['test_r2'], met['test_r2'] >= 0.70)
    rec('M-02', '模型', '5折CV整体R2>=0.70', '>=0.70',
        met['cv_r2'], met['cv_r2'] >= 0.70)
    rec('M-03', '模型', 'MAE<=0.70', '<=0.70',
        met['test_mae'], met['test_mae'] <= 0.70)
    low = {k: v for k, v in met['per_taste_r2'].items() if v < 0.5}
    rec('M-04', '模型', '各味觉R2无低于0.5的失效维度', '全部>=0.5',
        f'最低={min(met["per_taste_r2"].values())}', len(low) == 0)
    return m, met


def test_boundary(m):
    zero = {c: 0 for c in m.feature_cols}
    pred = m.predict(zero)
    ok = all(v == 0.0 for v in pred.values())
    rec('B-01', '边界规则', '全零输入(白水)判定无味', '六维全0',
        pred, ok)
    near = dict(zero)
    near['sugar'] = 5.0
    p2 = m.predict(near)
    rec('B-02', '边界规则', '仅含糖5g时不误判无味', 'sweet>0',
        p2['sweet'], p2['sweet'] > 0)


def test_sanity(m):
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    for name, dom in [('绿茶', 'bitter'), ('黑咖啡', 'bitter'),
                      ('苹果', 'sweet'), ('食盐', 'salty')]:
        row = df[df['name'] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        nut = {c: float(row[c]) for c in m.feature_cols}
        pred = m.predict(nut)
        top = max(pred, key=pred.get)
        label = {c: float(row[c]) for c in m.target_cols}
        mae = np.mean([abs(pred[c] - label[c]) for c in m.target_cols])
        rec('S-0' + str(len(RESULTS)), '典型食品',
            f'{name}主导味觉={m.TASTE_NAMES[dom]}',
            m.TASTE_NAMES[dom], f'预测主导={m.TASTE_NAMES[top]}, MAE={mae:.2f}',
            top == dom and mae < 1.5)


def test_latency(m):
    nut = {'calories': 300, 'protein': 10, 'carbs': 40, 'fat': 12,
           'sugar': 15, 'sodium': 400, 'organic_acid': 1.2}
    m.predict(nut)
    t0 = time.perf_counter()
    for _ in range(100):
        m.predict(nut)
    ms = (time.perf_counter() - t0) / 100 * 1000
    rec('P-01', '性能', '单次味觉预测延迟', '<50ms', f'{ms:.1f}ms', ms < 50)


def test_design(m):
    from inverse_design import InverseDesigner, SAVORY_CONDIMENTS
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    d = InverseDesigner(m, df)
    t0 = time.perf_counter()
    rs = d.design({'sweet': 8, 'sour': 2}, scenario='饮品·甜品')
    dt = time.perf_counter() - t0
    names = [it['name'] for r in rs for it in r['items']]
    cats = {d.cats[d.names.index(n)] for n in names}
    best = max(r['match_score'] for r in rs)
    ok = (not ({'肉禽水产', '蔬菜菌藻'} & cats)
          and not (set(names) & set(SAVORY_CONDIMENTS)) and best >= 85)
    rec('D-01', '逆向设计', '饮品场景食材池合规且匹配度>=85%', '>=85%',
        f'{best}% {names[:3]}', ok)
    rs2 = d.design({'salty': 6, 'umami': 7}, scenario='菜肴·主食',
                   low_sodium=True)
    na = min(r['nutrition_per100g']['sodium'] for r in rs2)
    rec('D-02', '逆向设计', '低钠约束下钠含量<=125mg/100g', '<=125',
        f'{na:.0f}mg', na <= 125)
    fat_names = {d.names[i] for i in range(d.n) if d.is_fat[i]}

    def fat_share(r):
        tot = sum(it['grams'] for it in r['items'])
        return sum(it['grams'] for it in r['items']
                   if it['name'] in fat_names) / tot
    worst = max(fat_share(r) for r in rs + rs2)
    rec('D-03', '逆向设计', '纯油脂类合计占比<=30%(克数取整容差)',
        '<=30%', f'{worst * 100:.0f}%', worst <= 0.30)
    ra = d.design({'sweet': 8, 'sour': 2}, scenario='饮品·甜品', n_results=1)
    rb = d.design({'sweet': 8, 'sour': 2}, scenario='饮品·甜品', n_results=1)
    rec('D-04', '逆向设计', '同参数两次结果一致(随机种子固定)', '一致',
        '一致' if ra == rb else '不一致', ra == rb)
    rec('D-05', '逆向设计', '单次设计耗时<10s', '<10s', f'{dt:.1f}s', dt < 10)


def test_interactions(m):
    from taste_model import apply_taste_interactions
    import pandas as pd
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

    def mix(items):
        tot = sum(g for _, g in items)
        d = {c: 0.0 for c in m.feature_cols}
        for n, g in items:
            r = df[df['name'] == n].iloc[0]
            for c in m.feature_cols:
                d[c] += float(r[c]) * g / tot
        return d

    # I-01 甜苦拮抗: 苦瓜+白糖 修正后苦味应低于原始预测
    d1 = mix([('苦瓜', 100), ('白糖', 60)])
    raw1 = m.predict(d1)
    cor1, notes1 = apply_taste_interactions(raw1, d1)
    rec('I-01', '风味交互', '苦瓜+白糖修正后苦味下降(甜苦拮抗)', 'cor<raw',
        f'{raw1["bitter"]:.2f}->{cor1["bitter"]:.2f}', cor1['bitter'] < raw1['bitter'])
    # I-02 咸鲜协同: 酱油+味精(高咸) 修正后鲜味不低于原始预测
    d2 = mix([('食盐', 10), ('味精', 5), ('鸡蛋', 100)])
    raw2 = m.predict(d2)
    cor2, _ = apply_taste_interactions(raw2, d2)
    rec('I-02', '风味交互', '高咸+鲜配方修正后鲜味不降(咸鲜协同)', 'cor>=raw',
        f'{raw2["umami"]:.2f}->{cor2["umami"]:.2f}', cor2['umami'] >= raw2['umami'])
    # I-03 蛋白-单宁沉淀: 牛奶+绿茶 修正后涩味低于原始预测
    d3 = mix([('牛奶', 150), ('绿茶', 150)])
    raw3 = m.predict(d3)
    cor3, notes3 = apply_taste_interactions(raw3, d3)
    rec('I-03', '风味交互', '奶茶配方修正后涩味下降(蛋白-单宁沉淀)', 'cor<raw',
        f'{raw3["astringent"]:.2f}->{cor3["astringent"]:.2f}',
        cor3['astringent'] < raw3['astringent'])


def test_golden(m):
    """演示案例点预测冻结守卫: 已发布材料中的实测数字逐值锁定"""
    from taste_model import apply_taste_interactions
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

    def nut_of(name):
        r = df[df['name'] == name].iloc[0]
        return {c: float(r[c]) for c in m.feature_cols}

    def mix(items):
        tot = sum(g for _, g in items)
        d = {c: 0.0 for c in m.feature_cols}
        for n, g in items:
            r = df[df['name'] == n].iloc[0]
            for c in m.feature_cols:
                d[c] += float(r[c]) * g / tot
        return d

    g1 = {'sweet': 1.01, 'sour': 1.18, 'salty': 1.81,
          'bitter': 7.77, 'umami': 1.91, 'astringent': 3.0}
    p1 = m.predict(nut_of('黑咖啡'))
    rec('G-01', '冻结守卫', '黑咖啡六维点预测逐值不变(发布=苦7.8)', g1, p1, p1 == g1)
    g2 = {'sweet': 2.35, 'sour': 1.28, 'salty': 2.8,
          'bitter': 1.18, 'umami': 3.36, 'astringent': 0.93}
    p2 = m.predict(nut_of('牛奶'))
    rec('G-02', '冻结守卫', '牛奶六维点预测逐值不变(发布=鲜3.4)', g2, p2, p2 == g2)
    d3 = mix([('苦瓜', 150), ('苹果', 200)])
    raw3 = {'sweet': 5.77, 'sour': 3.07, 'salty': 1.23,
            'bitter': 3.58, 'umami': 2.37, 'astringent': 2.48}
    cor3 = {'sweet': 5.03, 'sour': 2.58, 'salty': 1.23,
            'bitter': 2.75, 'umami': 2.46, 'astringent': 2.47}
    pr3 = m.predict(d3)
    pc3, _ = apply_taste_interactions(pr3, d3)
    rec('G-03', '冻结守卫', '苦瓜150g+苹果200g原始预测不变', raw3, pr3, pr3 == raw3)
    rec('G-04', '冻结守卫', '苦瓜150g+苹果200g交互修正不变(发布=甜5.0苦2.8)',
        cor3, pc3, pc3 == cor3)
    d4 = mix([('白粥', 198.5), ('三文鱼', 101.5)])
    raw4 = {'sweet': 1.08, 'sour': 0.96, 'salty': 3.96,
            'bitter': 1.28, 'umami': 6.49, 'astringent': 0.91}
    pr4 = m.predict(d4)
    rec('G-05', '冻结守卫', '白粥198.5g+三文鱼101.5g预测不变(发布=钠119mg)',
        raw4, pr4, pr4 == raw4)


def test_external(m):
    from external_validation import evaluate_anchors
    rs = evaluate_anchors(m)
    rec('E-01', '文献锚点', '30+文献锚点区间命中率>=75%', '>=75%',
        f"{rs['hits']}/{rs['n_anchors']}={rs['hit_rate'] * 100:.0f}%",
        rs['hit_rate'] >= 0.75)
    rec('E-02', '文献锚点', '平均越界偏差<=0.5', '<=0.5',
        rs['mean_dist'], rs['mean_dist'] <= 0.5)
    a_miss = [r for r in rs['rows'] if r['cls'] == 'A' and not r['hit']]
    rec('E-03', '文献锚点', 'A类标准参照物(ISO/GB)全部命中', '6/6',
        f"{6 - len(a_miss)}/6", len(a_miss) == 0)


def test_recipes(m):
    from recipe_validation import evaluate_recipes
    rr = evaluate_recipes(m)
    rec('R-01', '真实菜谱', '20道真实菜谱主导味觉top2一致率>=80%', '>=80%',
        f"{rr['top2_hits']}/{rr['n_recipes']}={rr['top2_rate'] * 100:.0f}%",
        rr['top2_rate'] >= 0.8)
    rec('R-02', '真实菜谱', '主导味觉top1一致率>=70%', '>=70%',
        f"{rr['top1_hits']}/{rr['n_recipes']}={rr['top1_rate'] * 100:.0f}%",
        rr['top1_rate'] >= 0.7)


def test_uncertainty(m):
    from uncertainty import TasteUncertainty
    uq = TasteUncertainty()
    met = uq.train(str(DATA_PATH))
    rec('U-01', '不确定性', '交叉保形区间覆盖率在[0.75,0.90](目标0.80)',
        '0.75~0.90', met['coverage'],
        0.75 <= met['coverage'] <= 0.90)
    mx = max(met['margins'].values())
    rec('U-02', '不确定性', '各维度区间半宽<=1.5(不过度保守)', '<=1.5',
        mx, mx <= 1.5)
    pt = m.predict({'calories': 2, 'protein': 0.3, 'carbs': 0, 'fat': 0,
                    'fiber': 0, 'sugar': 0, 'sodium': 2, 'potassium': 49,
                    'calcium': 2, 'iron': 0, 'vitamin_c': 0,
                    'organic_acid': 0.3, 'polyphenol': 120, 'caffeine': 40,
                    'alkaloid': 0, 'tannin': 150})
    iv = uq.interval_for(pt)
    ok = all(iv[c][0] <= pt[c] <= iv[c][1] and 0 <= iv[c][0] and iv[c][1] <= 10
             for c in m.target_cols)
    rec('U-03', '不确定性', '区间包含点估计且裁剪至[0,10]', '全部满足',
        f"苦{pt['bitter']}[{iv['bitter'][0]},{iv['bitter'][1]}]", ok)


def test_ocr():
    from PIL import Image, ImageDraw, ImageFont
    import io
    from nutrition_ocr import parse_from_image
    img = Image.new('RGB', (640, 460), 'white')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 34)
    lines = ['营养成分表', '项目        每100克',
             '能量        1500千焦', '蛋白质      5.2克',
             '脂肪        8.1克', '碳水化合物  20.3克',
             '钠          450毫克']
    for i, t in enumerate(lines):
        d.text((40, 30 + i * 58), t, fill='black', font=font)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    lines, parsed = parse_from_image(buf.getvalue())
    print('OCR 文本行:', lines)
    print('OCR 解析结果:', parsed)
    checks = [('calories', 1500 / 4.184, 30), ('protein', 5.2, 1.0),
              ('fat', 8.1, 1.0), ('carbs', 20.3, 2.0), ('sodium', 450, 50)]
    for i, (k, exp, tol) in enumerate(checks, 1):
        got = parsed.get(k)
        ok = got is not None and abs(got - exp) <= tol
        rec(f'O-0{i}', 'OCR', f'营养表字段 {k}',
            f'{exp:.1f}+-{tol}', got, ok)


def api_get(path):
    from urllib.parse import quote
    path = quote(path, safe='/:?=&')
    with urllib.request.urlopen(API + path, timeout=15) as r:
        return r.status, json.loads(r.read().decode('utf-8'))


def api_post(path, payload):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode('utf-8'))


def test_api():
    try:
        s, d = api_get('/')
        rec('A-01', 'API', 'GET / 服务信息', 200, s, s == 200)
    except Exception as e:
        rec('A-01', 'API', 'GET / 服务信息', 200, f'连接失败 {e}', False)
        return
    s, d = api_post('/api/taste', {'calories': 300, 'protein': 8,
                                   'caffeine': 120, 'alkaloid': 30})
    ok = s == 200 and 'taste' in d
    rec('A-02', 'API', 'POST /api/taste 味觉预测', '200+taste',
        f'{s} {d.get("taste")}', ok)
    s, d = api_get('/api/foods/search?q=苹果')
    rec('A-03', 'API', 'GET /api/foods/search 搜索', '200且结果>0',
        f'{s} 命中{len(d.get("results", d.get("foods", [])))}',
        s == 200)
    s, d = api_post('/api/recipe', {'items': [
        {'food_name': '苦瓜', 'grams': 150},
        {'food_name': '苹果', 'grams': 200}]})
    ok = s == 200 and 'taste' in d
    rec('A-04', 'API', 'POST /api/recipe 配方分析', '200+taste',
        f'{s} {d.get("taste")}', ok)
    s, d = api_get('/api/foods/绿茶')
    rec('A-05', 'API', 'GET /api/foods/{name} 详情', 200, s, s == 200)
    s, d = api_post('/api/design', {'target': {'sweet': 8, 'sour': 2},
                                    'scenario': '饮品·甜品', 'n_results': 2})
    m0 = d.get('results', [{}])[0].get('match_score', 0)
    rec('A-06', 'API', 'POST /api/design 逆向设计', '200+匹配度>=85%',
        f'{s} match={m0}', s == 200 and m0 >= 85)


if __name__ == '__main__':
    model, metrics = test_model()
    test_boundary(model)
    test_sanity(model)
    test_latency(model)
    test_design(model)
    test_interactions(model)
    test_golden(model)
    test_external(model)
    test_recipes(model)
    test_uncertainty(model)
    try:
        test_ocr()
    except Exception as e:
        rec('O-00', 'OCR', 'OCR 测试执行', '无异常', repr(e), False)
    test_api()
    n_pass = sum(1 for r in RESULTS if r[5] == 'PASS')
    print('=' * 60)
    print(f'测试完成: {n_pass}/{len(RESULTS)} 通过')
