# -*- coding: utf-8 -*-
"""
营养成分表 OCR 识别与解析模块
从食品包装"营养成分表"照片中提取文本，并解析为结构化营养数据。
使用 RapidOCR（ONNX 推理，纯 pip 安装，支持中文，无需外部二进制）。
"""
import io
import re

_engine = None


def _get_engine():
    """惰性加载 OCR 引擎（模型仅初始化一次）"""
    global _engine
    if _engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise RuntimeError(
                'OCR 依赖未安装，请先运行: pip install rapidocr_onnxruntime')
        _engine = RapidOCR()
    return _engine


def extract_text(image_bytes, scale=1):
    """从图片字节中提取文本行列表, scale>1 时先放大再识别。

    真实营养表含 NRV% 等多列, OCR 检测框顺序会交错(数值列先于
    字段名列返回), 直接按返回顺序拼接会导致"字段名-数值"失配。
    这里按检测框纵坐标聚类还原表格行, 行内按横坐标从左到右排序。
    """
    import numpy as np
    from PIL import Image
    engine = _get_engine()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale),
                         Image.LANCZOS)
    result, _ = engine(np.array(img))
    if not result:
        return []
    items = []
    heights = []
    for box, text, *_ in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((min(ys), min(xs), str(text)))
        heights.append(max(ys) - min(ys))
    items.sort()
    tol = (sorted(heights)[len(heights) // 2] * 0.6) if heights else 20
    rows = []
    for y, x, t in items:
        if rows and abs(rows[-1][0] - y) <= tol:
            rows[-1][1].append((x, t))
            rows[-1][0] = min(rows[-1][0], y)
        else:
            rows.append([y, [(x, t)]])
    lines = []
    for _, parts in rows:
        parts.sort()
        lines.append(' '.join(t for _, t in parts))
    return lines


# 字段解析规则：(字段名, 关键词正则, {单位: 换算系数})
# 标准单位与数据库一致：能量 kcal、宏量营养素 g、矿物质/维生素 mg
FIELD_RULES = [
    ('calories',  r'(?:能量|热量)',
     {'千焦': 1 / 4.184, 'kj': 1 / 4.184, '千卡': 1.0, '大卡': 1.0, 'kcal': 1.0}),
    ('protein',   r'蛋白质', {'克': 1.0, 'g': 1.0, '毫克': 0.001, 'mg': 0.001}),
    ('fat',       r'(?<!反式)(?<!饱和)(?<!不饱和)脂肪', {'克': 1.0, 'g': 1.0}),
    ('carbs',     r'碳水化合物', {'克': 1.0, 'g': 1.0}),
    ('sugar',     r'(?<![蔗乳果萄芽醇])糖', {'克': 1.0, 'g': 1.0}),
    ('fiber',     r'膳食纤维', {'克': 1.0, 'g': 1.0}),
    ('sodium',    r'(?<!碳酸氢)(?<!氯化)钠',
     {'毫克': 1.0, 'mg': 1.0, '克': 1000.0, 'g': 1000.0}),
    ('potassium', r'钾', {'毫克': 1.0, 'mg': 1.0}),
    ('calcium',   r'钙', {'毫克': 1.0, 'mg': 1.0}),
    ('iron',      r'铁', {'毫克': 1.0, 'mg': 1.0}),
    ('vitamin_c', r'(?:维生素\s*[CcＣ]|抗坏血酸)', {'毫克': 1.0, 'mg': 1.0}),
]

# 每100g 合理范围（超出视为误识别，丢弃）
SANITY = {
    'calories': (0, 900), 'protein': (0, 100), 'fat': (0, 100),
    'carbs': (0, 100), 'sugar': (0, 100), 'fiber': (0, 60),
    'sodium': (0, 40000), 'potassium': (0, 6000), 'calcium': (0, 3000),
    'iron': (0, 100), 'vitamin_c': (0, 2000),
}


def parse_nutrition(lines):
    """将 OCR 文本行解析为营养字典（字段名与数据库列一致）"""
    text = re.sub(r'\s+', ' ', ' '.join(lines))
    result = {}
    for field, keyword, units in FIELD_RULES:
        unit_pat = '|'.join(units)
        m = re.search(
            keyword + r'[^0-9]{0,12}?(\d+(?:\.\d+)?)\s*(' + unit_pat + r')',
            text, re.IGNORECASE)
        if not m:
            continue
        value = float(m.group(1)) * units[m.group(2).lower()]
        lo, hi = SANITY.get(field, (0, 1e9))
        if lo <= value <= hi:
            result[field] = round(value, 2)
    return result


# 判定文本确实是一张营养成分表(用于决定是否值得放大重试)
_TABLE_HINT = re.compile(r'(?:营养成分表|营养素参考值|能量|蛋白质)')


def parse_from_image(image_bytes):
    """端到端解析: OCR -> 结构化营养字典。

    单字字段名(钠/钾/钙/铁)在小图上易被检测器丢弃, 若确为营养表
    但钠字段缺失, 自动放大2倍重试一次并合并缺失字段。
    返回 (文本行列表, 营养字典)。
    """
    lines = extract_text(image_bytes)
    parsed = parse_nutrition(lines)
    joined = ' '.join(lines)
    if 'sodium' not in parsed and _TABLE_HINT.search(joined):
        lines2 = extract_text(image_bytes, scale=2)
        parsed2 = parse_nutrition(lines2)
        for k, v in parsed2.items():
            parsed.setdefault(k, v)
        if len(lines2) > len(lines):
            lines = lines2
            joined = ' '.join(lines)
    # 孤立毫克值归属: 国标 GB 28050 强制标识顺序为 能量/蛋白质/脂肪/
    # 碳水化合物/钠(1+4), 若核心四项齐全而钠缺失, 且文本中仅剩一个未被
    # 其他字段消费的毫克值, 判定其为钠(单字字段名易被OCR检测器丢弃)
    core = ('calories', 'protein', 'fat', 'carbs')
    if 'sodium' not in parsed and all(k in parsed for k in core):
        mg_vals = re.findall(r'(\d+(?:\.\d+)?)\s*(?:毫克|mg)', joined,
                             re.IGNORECASE)
        used = {round(parsed[k], 2) for k in
                ('potassium', 'calcium', 'iron', 'vitamin_c') if k in parsed}
        orphans = [float(v) for v in mg_vals
                   if round(float(v), 2) not in used]
        if len(orphans) == 1:
            lo, hi = SANITY['sodium']
            if lo <= orphans[0] <= hi:
                parsed['sodium'] = round(orphans[0], 2)
    return lines, parsed
