# -*- coding: utf-8 -*-
"""
味析 NutriTongue 技术方案 PDF 生成脚本
2026第八届全球校园人工智能算法精英大赛 - 赛题2: AI+软件创新
注意: 源码字符串内一律不使用中文弯引号，避免语法错误
"""
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, PageBreak,
                                HRFlowable, KeepTogether)
from reportlab.graphics.shapes import Drawing, String, Rect, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart

sys.path.insert(0, str(Path(__file__).parent))
from feature_importance import compute_importance, FEAT_CN, TASTE_CN  # noqa: E402

# 排列重要性归因矩阵(16特征 x 6味觉)，训练+分析约30秒
IMP, IMP_FEATS, IMP_TASTES = compute_importance(n_repeats=30)

# ---------------------------------------------------------------- 字体
try:
    # 官方模板格式: 正文宋体, 标题黑体(粗体)
    pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
    FONT, FONT_B = 'SimSun', 'SimHei'
except Exception:
    try:
        pdfmetrics.registerFont(TTFont('MSYH', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
        pdfmetrics.registerFont(TTFont('MSYHB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
        FONT, FONT_B = 'MSYH', 'MSYHB'
    except Exception:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        FONT = FONT_B = 'STSong-Light'

# ---------------------------------------------------------------- 颜色
ORANGE = HexColor('#f97316')
PINK = HexColor('#ec4899')
DARK = HexColor('#1f2937')
GRAY = HexColor('#6b7280')
LIGHT = HexColor('#fff7ed')
GREEN = HexColor('#22c55e')
BLUE = HexColor('#3b82f6')
TASTE_COLORS = [PINK, HexColor('#eab308'), BLUE, HexColor('#92400e'),
                HexColor('#ef4444'), HexColor('#8b5cf6')]

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / 'docs'
DOCS_DIR.mkdir(exist_ok=True)
OUT = DOCS_DIR / '味析NutriTongue技术方案.pdf'

# ---------------------------------------------------------------- 样式
ss = {}
ss['title'] = ParagraphStyle('title', fontName=FONT_B, fontSize=30, leading=42,
                             textColor=ORANGE, alignment=TA_CENTER, spaceAfter=10)
ss['subtitle'] = ParagraphStyle('subtitle', fontName=FONT, fontSize=14, leading=22,
                                textColor=GRAY, alignment=TA_CENTER, spaceAfter=6)
ss['cover_meta'] = ParagraphStyle('cover_meta', fontName=FONT, fontSize=11, leading=20,
                                  textColor=DARK, alignment=TA_CENTER)
ss['h1'] = ParagraphStyle('h1', fontName=FONT_B, fontSize=16, leading=26,
                          textColor=ORANGE, spaceBefore=14, spaceAfter=8)
ss['h2'] = ParagraphStyle('h2', fontName=FONT_B, fontSize=14, leading=22,
                          textColor=DARK, spaceBefore=8, spaceAfter=4)
ss['body'] = ParagraphStyle('body', fontName=FONT, fontSize=12, leading=17,
                            textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=5,
                            firstLineIndent=24)
ss['body_ni'] = ParagraphStyle('body_ni', parent=ss['body'], firstLineIndent=0)
ss['bullet'] = ParagraphStyle('bullet', fontName=FONT, fontSize=12, leading=17,
                              textColor=DARK, leftIndent=16, spaceAfter=3)
ss['small'] = ParagraphStyle('small', fontName=FONT, fontSize=9, leading=14,
                             textColor=GRAY)
ss['code'] = ParagraphStyle('code', fontName=FONT, fontSize=9,
                            leading=12.5, textColor=DARK,
                            backColor=HexColor('#f6f8fa'), leftIndent=8,
                            rightIndent=8, spaceBefore=2, spaceAfter=2)
ss['table_h'] = ParagraphStyle('table_h', fontName=FONT_B, fontSize=10, leading=14,
                               textColor=colors.white, alignment=TA_CENTER)
ss['table_c'] = ParagraphStyle('table_c', fontName=FONT, fontSize=9.5, leading=13,
                               textColor=DARK, alignment=TA_CENTER)
ss['table_l'] = ParagraphStyle('table_l', fontName=FONT,  fontSize=9.5, leading=13,
                               textColor=DARK, alignment=TA_LEFT)


def P(text, style='body'):
    return Paragraph(text, ss[style])


def H(num, text):
    """标题: '一、'/'附录A' 为一级(三号粗), '1.1' 为二级(四号粗); 并登记目录"""
    lvl = 1 if ('.' in num) else 0
    p = P(f'{num}　{text}', 'h2' if lvl else 'h1')
    p._toc = (lvl, f'{num}　{text}')
    return p


def std_table(data, col_widths, header_bg=ORANGE, zebra=True):
    # 页宽自适应: 模板页边距左右3cm, 可用宽15cm, 超宽表格等比收缩
    avail = 15 * cm
    total = sum(col_widths)
    if total > avail:
        col_widths = [w * avail / total for w in col_widths]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(('BACKGROUND', (0, r), (-1, r), HexColor('#f9fafb')))
    tbl.setStyle(TableStyle(style))
    return tbl


# ---------------------------------------------------------------- 图表
def r2_bar_chart():
    d = Drawing(425, 230)
    bc = VerticalBarChart()
    bc.x, bc.y = 42, 30
    bc.height, bc.width = 175, 380
    bc.data = [[0.860, 0.844, 0.883, 0.713, 0.772, 0.609]]
    bc.categoryAxis.categoryNames = ['甜味', '酸味', '咸味', '苦味', '鲜味', '涩味']
    bc.categoryAxis.labels.fontName = FONT
    bc.categoryAxis.labels.fontSize = 10
    bc.valueAxis.valueMin, bc.valueAxis.valueMax = 0, 1.0
    bc.valueAxis.valueStep = 0.2
    bc.valueAxis.labels.fontName = FONT
    bc.valueAxis.labels.fontSize = 8
    bc.barWidth = 16
    for i in range(6):
        bc.bars[(0, i)].fillColor = TASTE_COLORS[i]
    bc.barLabels.fontName = FONT
    bc.barLabels.fontSize = 9
    bc.barLabelFormat = '%.3f'
    bc.barLabels.dx = 0
    bc.barLabels.dy = 4
    d.add(bc)
    d.add(String(212, 215, '六维味觉留出测试 R2', fontName=FONT_B, fontSize=11,
                 textAnchor='middle', fillColor=DARK))
    return d


def compare_chart():
    d = Drawing(425, 230)
    bc = VerticalBarChart()
    bc.x, bc.y = 42, 30
    bc.height, bc.width = 175, 380
    bc.data = [[0.845, 0.890, 0.880, 0.407, 0.774, 0.567],
               [0.860, 0.844, 0.883, 0.713, 0.772, 0.609]]
    bc.categoryAxis.categoryNames = ['甜味', '酸味', '咸味', '苦味', '鲜味', '涩味']
    bc.categoryAxis.labels.fontName = FONT
    bc.categoryAxis.labels.fontSize = 10
    bc.valueAxis.valueMin, bc.valueAxis.valueMax = 0, 1.0
    bc.valueAxis.valueStep = 0.2
    bc.valueAxis.labels.fontName = FONT
    bc.valueAxis.labels.fontSize = 8
    bc.bars[0].fillColor = HexColor('#cbd5e1')
    bc.bars[1].fillColor = ORANGE
    bc.barWidth = 8
    d.add(bc)
    d.add(String(212, 215, '13特征(灰) vs 16特征(橙) 各味觉 R2 对比', fontName=FONT_B,
                 fontSize=11, textAnchor='middle', fillColor=DARK))
    return d


def importance_heatmap():
    """16特征 x 6味觉 排列重要性热力图(列内归一化着色, 单元格为原值)"""
    feats = [FEAT_CN[f] for f in IMP_FEATS]
    tastes = [TASTE_CN[t] for t in IMP_TASTES]
    cw, ch = 56, 15
    left, bottom = 66, 14
    W, H = left + cw * 6 + 10, bottom + ch * 16 + 34
    d = Drawing(W, H)
    col_max = IMP.max(axis=0)
    col_max[col_max <= 0] = 1
    for i in range(16):
        y = bottom + (15 - i) * ch
        d.add(String(left - 5, y + 4, feats[i], fontName=FONT, fontSize=7.5,
                     textAnchor='end', fillColor=DARK))
        for j in range(6):
            v = float(IMP[i, j])
            t = v / float(col_max[j])
            fc = Color(1, 1 - 0.55 * t, 1 - 0.82 * t)  # 白 -> 橙
            d.add(Rect(left + j * cw, y, cw, ch, fillColor=fc,
                       strokeColor=colors.white, strokeWidth=0.6))
            if v >= 0.005:
                tc = colors.white if t > 0.55 else DARK
                d.add(String(left + j * cw + cw / 2, y + 4, f'{v:.2f}',
                             fontName=FONT, fontSize=6.8,
                             textAnchor='middle', fillColor=tc))
    for j in range(6):
        d.add(String(left + j * cw + cw / 2, bottom + 16 * ch + 8, tastes[j],
                     fontName=FONT_B, fontSize=8.5,
                     textAnchor='middle', fillColor=DARK))
    d.add(String(left + cw * 3, bottom + 16 * ch + 22,
                 '排列重要性热力图(数值为打乱该特征后 R2 的下降量, 颜色按列归一化)',
                 fontName=FONT_B, fontSize=10.5,
                 textAnchor='middle', fillColor=DARK))
    return d


def arch_diagram():
    """系统架构: 感知层 -> 服务层 -> 应用层"""
    d = Drawing(425, 108)
    layers = [
        ('感知层(数据输入)', ['营养成分表拍照OCR', '手动输入16项指标', '281种食品数据库']),
        ('服务层(味觉计算)', ['FastAPI 标准接口', 'AI模型推理引擎', '营养分析与健康评分']),
        ('应用层(价值交付)', ['Streamlit Web应用', '配方/对比/预测', '第三方系统集成']),
    ]
    bw, bh, gap = 125, 88, 21
    for k, (title, items) in enumerate(layers):
        x = 4 + k * (bw + gap)
        d.add(Rect(x, 8, bw, bh, rx=8, ry=8,
                   fillColor=LIGHT, strokeColor=ORANGE, strokeWidth=1.2))
        d.add(String(x + bw / 2, 8 + bh - 16, title, fontName=FONT_B,
                     fontSize=9.5, textAnchor='middle', fillColor=ORANGE))
        for m, it in enumerate(items):
            d.add(String(x + bw / 2, 8 + bh - 34 - m * 16, it,
                         fontName=FONT, fontSize=8,
                         textAnchor='middle', fillColor=DARK))
        if k < 2:
            ax = x + bw + 4
            d.add(Line(ax, 52, ax + gap - 8, 52,
                       strokeColor=PINK, strokeWidth=2))
            d.add(Line(ax + gap - 12, 56, ax + gap - 8, 52,
                       strokeColor=PINK, strokeWidth=2))
            d.add(Line(ax + gap - 12, 48, ax + gap - 8, 52,
                       strokeColor=PINK, strokeWidth=2))
    return d


# ---------------------------------------------------------------- 页眉页脚
def on_page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFillColor(ORANGE)
        canvas.rect(0, A4[1] - 8, A4[0], 8, fill=1, stroke=0)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(doc.leftMargin, 1.5 * cm,
                          '味析 NutriTongue - AI虚拟电子舌 技术方案')
        canvas.drawRightString(A4[0] - doc.rightMargin, 1.5 * cm,
                               f'第 {doc.page} 页')
    canvas.restoreState()


# ---------------------------------------------------------------- 文档
story = []

# 封面
story.append(Spacer(1, 3.2 * cm))
story.append(P('味析 · NutriTongue', 'title'))
story.append(P('AI 虚拟电子舌', 'subtitle'))
story.append(P('—— 基于营养化学特征与多输出梯度提升树的味觉预测系统 ——', 'subtitle'))
story.append(Spacer(1, 1.4 * cm))
cover_tbl = Table([
    ['参赛赛事', '2026第八届全球校园人工智能算法精英大赛 · 算法创新赛'],
    ['参赛团队', '味来科技'],
    ['参赛编号', 'AIC-2026-45228632'],
    ['参赛赛道', '赛题2: AI+软件创新'],
    ['作品类型', '纯软件方案(零硬件 · 零实验条件依赖)'],
    ['数据规模', '9大类 281 种食品 · 16维输入特征 · 6维味觉标注'],
    ['核心模型', '多输出梯度提升回归 + 数据增强 + 样本加权 + 五折交叉验证'],
    ['整体精度', '留出测试 R2 = 0.780 / 五折交叉验证 R2 = 0.810'],
    ['系统形态', 'Streamlit Web应用 + FastAPI 味觉计算API + 营养成分表OCR识别'],
    ['在线体验', 'https://nutritongue.streamlit.app (Streamlit Cloud 永久部署, 首次访问约30秒唤醒)'],
], colWidths=[3.2 * cm, 11.5 * cm])
cover_tbl.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 10.5),
    ('FONTNAME', (0, 0), (0, -1), FONT_B),
    ('TEXTCOLOR', (0, 0), (0, -1), ORANGE),
    ('BACKGROUND', (0, 0), (0, -1), LIGHT),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#fed7aa')),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('LEFTPADDING', (0, 0), (-1, -1), 10),
]))
story.append(cover_tbl)
story.append(Spacer(1, 1.8 * cm))
story.append(P('2026 年 10 月', 'cover_meta'))
story.append(Spacer(1, 0.4 * cm))
story.append(P('关键词: 电子舌 · 味觉预测 · 机器学习 · 梯度提升树 · 营养评价 · 风味化学',
               'cover_meta'))
story.append(PageBreak())

# ---------------------------------------------------------------- 目录页
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

story.append(Paragraph('目　录', ParagraphStyle(
    'toc_title', fontName=FONT_B, fontSize=22, leading=36,
    alignment=TA_CENTER, spaceAfter=16)))
_toc = TableOfContents()
_toc.levelStyles = [
    ParagraphStyle('toc_l0', fontName=FONT_B, fontSize=12, leading=22,
                   leftIndent=4, textColor=DARK),
    ParagraphStyle('toc_l1', fontName=FONT, fontSize=10.5, leading=17,
                   leftIndent=26, textColor=DARK),
]
story.append(_toc)
story.append(PageBreak())

# ---------------------------------------------------------------- 作品简介(模板独立部分, <=300字)
_ip = P('作品简介', 'h1')
_ip._toc = (0, '作品简介')
story.append(_ip)
story.append(P('味析NutriTongue是一款纯软件AI虚拟电子舌：无需任何传感器，'
               '输入食品16项营养与风味化学指标，多输出梯度提升模型即可预测'
               '甜、酸、咸、苦、鲜、涩六维味觉强度（281种食品自建库，'
               '测试集R2=0.780，交叉验证0.810）。系统覆盖"查、尝、配、反设计"'
               '完整链路：食品查询、拍照OCR识别营养表、配方风味交互分析，'
               '并可将模型反向使用——输入目标味觉，一秒内搜索出可烹饪配方，'
               '支持低钠低糖约束。'))
story.append(P('模型经归因验证符合风味化学规律，预测附带80%保形置信区间，'
               '并通过32项文献锚点与20道真实菜谱外部验证；能力封装为开放API。'
               '面向低糖低钠配方研发场景，已完成22人真实试用'
               '（主导味觉一致率82.5%）。让AI像人一样尝味道，'
               '用数据为健康饮食决策服务。'))
story.append(PageBreak())

# 一 项目概述
story.append(H('一、', '项目概述'))
story.append(H('1.1', '项目背景与意义: 电子舌技术及其硬件门槛'))
story.append(P('电子舌(E-tongue)是一类模拟人类味觉系统的分析仪器，通过物理电化学传感器阵列'
               '获取液体样本的响应信号，再经模式识别算法区分甜、酸、咸、苦、鲜等基本味觉。'
               '传统电子舌存在三方面门槛: 一是传感器阵列价格昂贵、维护成本高；'
               '二是必须取得真实食品样本并完成前处理与测量实验；'
               '三是传感器漂移、交叉敏感性强，标定复杂。对于没有实验室与硬件条件的研究者，'
               '基于物理电子舌的创新几乎无法开展。'))
story.append(H('1.2', '核心思路: 把味觉问题转化为数据回归问题'))
story.append(P('现代食品化学已明确: 味觉感受由食品中的具体化学物质决定。例如甜味主要来自'
               '糖类等甜味物质，咸味与钠离子浓度高度相关，酸味由柠檬酸、苹果酸等有机酸驱动，'
               '苦味与多酚、生物碱相关，鲜味来自氨基酸与核苷酸，涩味则与多酚-蛋白质'
               '收敛作用有关。既然味觉有其物质基础，就可以绕开物理传感器，'
               '直接学习「可测量营养/化学指标 → 六维味觉强度」的映射关系，'
               '用纯软件方式实现电子舌的核心功能。'))
story.append(H('1.3', '赛题方向定位与作品定义'))
story.append(P('味析(NutriTongue)是一款AI虚拟电子舌系统: 输入一份食品(或任意食材配方)'
               '每100克的16项营养与化学指标，系统即可输出该食品在甜、酸、咸、苦、鲜、涩'
               '六个维度上的味觉强度(0-10分)，并同步给出营养健康评分、风味文字描述与'
               '搭配建议。系统以交互式Web应用形式交付，零硬件成本、可复制、易推广，'
               '面向低糖低钠配方研发、预制菜口味质控、慢病膳食管理与食品专业教学'
               '四类真实场景(详见5.5节)。'))

# 二 数据方案
story.append(H('二、', '数据库构建'))
story.append(H('2.1', '总体结构'))
story.append(P('自建结构化食品数据库共 281 种食品，覆盖 9 大日常饮食类别，'
               '每条样本包含 25 个字段: 3 个标识字段(编号、名称、类别)、'
               '16 个输入特征字段、6 个味觉标签字段。所有营养指标统一为每100g可食部含量，'
               '味觉标签为0-10的整数人工标注(0表示无此味，10表示该味极强)。'))
cat_data = [
    [P('类别', 'table_h'), P('数量(种)', 'table_h'), P('典型食品', 'table_h')],
    [P('水果', 'table_c'), P('40', 'table_c'), P('苹果、香蕉、柠檬、山楂、百香果', 'table_l')],
    [P('蔬菜菌藻', 'table_c'), P('45', 'table_c'), P('西红柿、苦瓜、海带、香菇、青椒', 'table_l')],
    [P('肉禽水产', 'table_c'), P('40', 'table_c'), P('牛肉、鸡胸肉、三文鱼、虾、猪肝', 'table_l')],
    [P('主食', 'table_c'), P('30', 'table_c'), P('米饭、馒头、燕麦、红薯、全麦面包', 'table_l')],
    [P('饮料酒水', 'table_c'), P('30', 'table_c'), P('黑咖啡、绿茶、可乐、啤酒、柠檬水', 'table_l')],
    [P('零食加工', 'table_c'), P('30', 'table_c'), P('黑巧克力、薯片、蛋糕、牛肉干、果冻', 'table_l')],
    [P('调味品', 'table_c'), P('25', 'table_c'), P('酱油、食醋、食盐、蜂蜜、番茄酱', 'table_l')],
    [P('蛋奶豆', 'table_c'), P('20', 'table_c'), P('鸡蛋、牛奶、豆腐、奶酪、豆浆', 'table_l')],
    [P('坚果种子', 'table_c'), P('20', 'table_c'), P('核桃、杏仁、黑芝麻、花生、亚麻籽', 'table_l')],
]
story.append(std_table(cat_data, [2.6 * cm, 2.0 * cm, 10.1 * cm]))
story.append(Spacer(1, 4 * mm))

story.append(H('2.2', '16维输入特征与科学依据'))
feat_data = [
    [P('特征', 'table_h'), P('单位', 'table_h'), P('味觉/营养意义', 'table_h')],
    [P('热量', 'table_l'), P('kcal', 'table_c'), P('综合能量水平，影响风味浓郁度', 'table_l')],
    [P('蛋白质', 'table_l'), P('g', 'table_c'), P('鲜味前体物质(氨基酸/肽)的近似指标', 'table_l')],
    [P('碳水化合物', 'table_l'), P('g', 'table_c'), P('主食特征，与基础味型相关', 'table_l')],
    [P('脂肪', 'table_l'), P('g', 'table_c'), P('脂香载体，可缓解涩感、增强醇厚感', 'table_l')],
    [P('膳食纤维', 'table_l'), P('g', 'table_c'), P('植物性食品标志，关联粗糙/收敛口感', 'table_l')],
    [P('糖分', 'table_l'), P('g', 'table_c'), P('甜味最主要的物质基础', 'table_l')],
    [P('钠', 'table_l'), P('mg', 'table_c'), P('咸味直接来源(食盐为氯化钠)', 'table_l')],
    [P('钾 / 钙 / 铁', 'table_l'), P('mg', 'table_c'), P('矿物质背景，参与咸苦味平衡', 'table_l')],
    [P('维生素C', 'table_l'), P('mg', 'table_c'), P('果蔬标志，常见于酸甜味食品', 'table_l')],
    [P('有机酸(一期新增)', 'table_l'), P('g', 'table_c'),
     P('柠檬酸/苹果酸/酒石酸等总量，酸味的直接物质基础', 'table_l')],
    [P('多酚(一期新增)', 'table_l'), P('mg', 'table_c'),
     P('茶多酚/花青素/黄酮等总量，苦味与涩味的重要来源', 'table_l')],
    [P('咖啡因(二期新增)', 'table_l'), P('mg', 'table_c'),
     P('甲基黄嘌呤类，咖啡/茶/巧克力特有苦味来源', 'table_l')],
    [P('苦味生物碱(二期新增)', 'table_l'), P('mg', 'table_c'),
     P('胡椒碱/苦瓜苷/花椒酰胺等非咖啡因苦味生物碱总量', 'table_l')],
    [P('单宁(二期新增)', 'table_l'), P('mg', 'table_c'),
     P('缩合与水解单宁，茶/红酒/柿子苦涩味来源，与总多酚区分口径', 'table_l')],
]
story.append(std_table(feat_data, [3.4 * cm, 1.6 * cm, 9.7 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(P('特征升级动机: 在早期11特征实验中，酸味、苦味、涩味三个维度的预测 R2 长期偏低'
               '(酸味0.64、苦味0.41、涩味0.56)。基于味觉生理学分析，根本原因是缺失了'
               '决定这三类味觉的直接物质变量。一期补充有机酸、多酚后酸味 R2 跃升至0.890；'
               '二期针对苦味继续补充咖啡因、苦味生物碱、单宁三个化学特征，'
               '使苦味 R2 由0.407提升至0.619(二期实测)，模型迭代后进一步提升至0.713，形成16维输入。', 'body'))
story.append(H('2.3', '数据质量控制'))
for txt in [
    '统一口径: 全部指标按每100g可食部计，单位在字段层面固定；',
    '程序校验: 使用脚本验证总行数(281)、字段数(25)、空值(0)、编号唯一性与名称唯一性；',
    '极值核查: 保留食盐(钠38758mg)等真实极值，使模型覆盖调味品等极端分布；',
    '类别均衡: 9大类样本量在20-45之间，按类别分层划分训练/测试集，避免小类泄漏。',
]:
    story.append(Paragraph('• ' + txt, ss['bullet']))

story.append(PageBreak())

# 三 算法
story.append(H('三、', '算法方案'))
story.append(H('3.1', '任务建模'))
story.append(P('将味觉预测建模为多输出回归任务: 输入x为16维特征向量，'
               '输出y为6维味觉强度向量。各味觉维度共享同一组输入但取值规律不同，'
               '采用 MultiOutputRegressor 为每个味觉维度独立训练一个'
               ' GradientBoostingRegressor(梯度提升回归树)，'
               '兼顾非线性拟合能力、训练效率与可解释性。'))
story.append(H('3.2', '训练流程'))
for txt in [
    '标准化: 对16维特征做Z-Score标准化，消除量纲差异(钠跨度达4个数量级)；',
    '数据增强: 仅对训练集施加3轮高斯乘性噪声(标准差2%)，扩充至4倍样本，提升鲁棒性；',
    '样本加权: 针对苦味长尾分布，对苦味≥5的样本加权2.5倍、≥7的样本加权3.0倍，'
    '缓解模型被低苦味样本主导的问题；',
    '分层留出: 按食品类别(stratify=category)留出20%(57种)作为独立测试集；',
    '部署再训练: 评估通过后使用全部281条样本重新训练最终模型；',
    '交叉验证: 另做5折交叉验证(KFold)，用 cross_val_predict 复核泛化稳定性；',
    '边界规则: 当所有味觉相关成分均为0(如白水)时直接判定无味，避免零向量落在训练分布外。',
]:
    story.append(Paragraph('• ' + txt, ss['bullet']))
story.append(H('3.3', '关键超参数'))
hp = [
    [P('参数', 'table_h'), P('取值', 'table_h'), P('说明', 'table_h')],
    [P('n_estimators', 'table_l'), P('200', 'table_c'), P('boosting树数量(经9组超参网格调优确定)', 'table_l')],
    [P('max_depth', 'table_l'), P('3', 'table_c'), P('单棵树最大深度，控制过拟合', 'table_l')],
    [P('learning_rate', 'table_l'), P('0.05', 'table_c'), P('学习率/收缩系数', 'table_l')],
    [P('subsample', 'table_l'), P('0.8', 'table_c'), P('随机梯度提升采样比，抑制小样本过拟合', 'table_l')],
    [P('loss', 'table_l'), P('squared_error', 'table_c'), P('平方损失，与R2评估口径一致', 'table_l')],
    [P('苦味样本权重', 'table_l'), P('2.5 / 3.0', 'table_c'), P('苦味>=5加权2.5倍、>=7加权3.0倍(长尾)', 'table_l')],
    [P('增强轮数', 'table_l'), P('3', 'table_c'), P('高斯噪声2%，训练集扩至896条', 'table_l')],
]
story.append(std_table(hp, [3.4 * cm, 2.0 * cm, 9.3 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(H('3.4', '营养评价与综合评价'))
story.append(P('营养分析引擎参考《中国居民膳食指南》每日推荐摄入量，计算各营养素占比；'
               '健康评分以70分为基准，对膳食纤维、优质蛋白、维生素C加分，'
               '对高糖、高钠、高脂扣分，映射为优秀/良好/一般/较差/很差五级。'
               '综合评价模块再结合六维味觉的丰富度与强度生成0-100综合风味分、'
               '主导味觉文字描述与风味搭配建议。'))
story.append(H('3.5', '模型可解释性: 特征归因分析'))
story.append(P('为验证16维特征设计的科学性，采用排列重要性(Permutation Importance)'
               '对六个味觉维度分别计算各输入特征的贡献度——即随机打乱某特征后'
               '模型R2的下降量，下降越多说明模型对该特征依赖越强。'
               '结果如下图所示，每个味觉维度的主导特征都恰好是其食品化学本质:'))
story.append(Spacer(1, 2 * mm))
story.append(importance_heatmap())
story.append(Spacer(1, 2 * mm))
story.append(P('归因结果呈现清晰的机理一致性: 甜味由糖分主导(1.34)、咸味由钠主导(1.51)、'
               '鲜味由蛋白质主导(1.02)、酸味由有机酸主导(1.04)，四个维度均为单一主导结构，'
               '与味觉生理机制严格对应，也解释了这四维R2普遍较高的原因；'
               '苦味前三位依次为苦味生物碱(0.62)、单宁(0.35)、咖啡因(0.09)，'
               '二期新增的三个化学特征全部进入主导位置，证明特征扩充直击靶心；'
               '涩味则呈多因素耦合结构(膳食纤维0.22、多酚0.14、单宁0.13、生物碱0.13)，'
               '缺少单一主导变量，这正是涩味R2(0.609)相对偏低的原因，也指明了后续优化方向。'
               '该分析表明模型学习到的是风味化学规律而非统计噪声，'
               'AI预测具备了可解释的科学与决策依据。'))

# 四 实验
story.append(H('3.6', '逆向配方设计: 从目标味觉反向搜索配方'))
story.append(P('正向模型解决「给配方、测味道」，逆向设计解决「给味道、找配方」。'
               '用户在六维味觉空间给出目标向量，可附加低钠(<=120mg/100g，GB 28050 '
               '低钠声称标准)、低糖(<=5g/100g)约束，并选择应用场景(通用/饮品·甜品/'
               '菜肴·主食)。系统按场景限定候选食材池(如饮品场景排除肉禽水产与咸口'
               '调味品)，随机采样数万组「食材组合 x Dirichlet随机配比」，'
               '用训练好的模型对混合营养做批量味觉预测，以目标欧氏距离叠加约束罚分'
               '(钠/糖超限、纯油脂类占比>25%、调味品占比超场景上限)排序，'
               '再对前列候选做批量坐标扰动精修，最后按食材重合度做多样性挑选，'
               '输出Top-3方案。全流程无需额外训练，随机种子固定可复现，'
               '单次设计约0.6秒。'))
story.append(P('典型案例: 目标「甜8酸2」饮品场景，系统输出「苹果汁191.9g+哈密瓜67.4g+白糖40.7g」，'
               '预测甜8.09/酸2.94，匹配度90.7%；目标「咸6鲜7」并开启低钠约束的'
               '菜肴场景，输出「白粥198.5g+三文鱼101.5g」，钠119mg/100g达标，匹配度87.7%。'
               '食材池自动排除可乐、薯片等预包装成品，仅保留可用于烹饪调配的基础原料，'
               '确保输出配方具备实际可操作性。'
               '每条方案可由用户选择「食品设计」或「饮食搭配」输出类型: '
               '前者输出单一成品(施加类别一致性罚分, 避免跨风味方向混搭), '
               '后者输出一餐内可独立成菜的食材组合(放宽类别一致性, 仅保留化学/烹饪依据的冲突罚分); '
               '味觉预测基于原料营养按克数加和计算，未建模烹饪过程(美拉德反应、焦糖化、蛋白质变性等)对风味的'
               '影响，生食类方案预测可信度较高，需烹饪类方案作为食材选型与配比参考。'
               '逆向设计与正向预测构成「分析-创造」能力闭环，'
               '直接对应食品研发中「低糖低钠配方定向设计」的真实工作流。'))
story.append(H('3.7', '风味交互修正层: 超越线性混合'))
story.append(P('配方按每100g营养加权平均后回归，得到的是「平均化学味」；而真实品尝中'
               '各味觉维度存在味觉生理心理学已证实的拮抗与协同效应。系统在配方预测之上'
               '叠加四条文献级交互规则: ①甜苦互抑(蔗糖抑制苦味感知，苦味亦轻度抑甜); '
               '②咸鲜协同(NaCl 增强鲜味，上限+20%); ③甜酸互抑(糖降低酸感); '
               '④蛋白-单宁沉淀(蛋白质结合单宁降低涩感，即奶茶原理)。'
               '每条规则幅度设上限、确定性可复现，并输出修正说明。'
               '专项测试: 苦瓜100g+白糖60g 预测苦味 3.45→2.22; '
               '高咸+鲜配方鲜味 7.24→8.54; 牛奶+绿茶涩感下降。'
               '此外配方分析页内置搭配合理性规则层: 检测到刺激性香辛料×高甜食材、'
               '血制品/内脏×水果(有机酸/单宁与铁、蛋白结合产生腥涩异味)、'
               '强酸×未发酵乳(酪蛋白酸变性凝结)三类冲突组合时明确提示并扣减综合评分，'
               '综合评分同时展示构成明细(风味分×0.6＋健康分×0.4－搭配扣减)，'
               '且丰富度按前4维每维10分、超出每维3分边际递减，避免「堆食材」虚高——'
               '「风味和谐度」不在六维标签空间内，'
               '无法靠增加训练数据习得，故以规则层显式处理(已知局限与展望)。'))
story.append(Spacer(1, 2 * mm))
story.append(H('四、', '实验结果与分析'))
story.append(H('4.1', '整体性能'))
m_tbl = [
    [P('指标', 'table_h'), P('13特征基线', 'table_h'), P('16特征(本作品)', 'table_h'),
     P('变化', 'table_h')],
    [P('留出测试 R2', 'table_l'), P('0.727', 'table_c'), P('0.780', 'table_c'),
     P('+0.053', 'table_c')],
    [P('5折交叉验证 R2', 'table_l'), P('0.772', 'table_c'), P('0.810', 'table_c'),
     P('+0.038', 'table_c')],
    [P('RMSE(均方根误差)', 'table_l'), P('0.974', 'table_c'), P('0.920', 'table_c'),
     P('-0.054', 'table_c')],
    [P('MAE(平均绝对误差)', 'table_l'), P('0.652', 'table_c'), P('0.615', 'table_c'),
     P('-0.037', 'table_c')],
]
story.append(std_table(m_tbl, [4.6 * cm, 3.2 * cm, 3.6 * cm, 2.6 * cm]))
story.append(Spacer(1, 4 * mm))
story.append(H('4.2', '分味觉维度对比'))
pt = [[P('味觉', 'table_h'), P('13特征 R2', 'table_h'), P('16特征 R2', 'table_h'),
       P('解读', 'table_h')],
      [P('甜味', 'table_c'), P('0.845', 'table_c'), P('0.860', 'table_c'),
       P('糖分主导，精度高', 'table_l')],
      [P('酸味', 'table_c'), P('0.890', 'table_c'), P('0.844', 'table_c'),
       P('有机酸入模，精度稳定(划分间小幅波动)', 'table_l')],
      [P('咸味', 'table_c'), P('0.880', 'table_c'), P('0.883', 'table_c'),
       P('钠特征直接，精度最高梯队', 'table_l')],
      [P('苦味', 'table_c'), P('0.407', 'table_c'), P('0.713', 'table_c'),
       P('苦味三特征入模+模型迭代，提升 +0.306', 'table_l')],
      [P('鲜味', 'table_c'), P('0.774', 'table_c'), P('0.772', 'table_c'),
       P('蛋白质近似，精度良好', 'table_l')],
      [P('涩味', 'table_c'), P('0.567', 'table_c'), P('0.609', 'table_c'),
       P('多酚与单宁提供一定增益', 'table_l')]]
story.append(std_table(pt, [1.8 * cm, 2.6 * cm, 2.6 * cm, 7.0 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(compare_chart())
story.append(Spacer(1, 2 * mm))
story.append(r2_bar_chart())
story.append(PageBreak())

story.append(H('4.3', '算法选型对比实验'))
story.append(P('在同一16维特征、同一类别分层留出划分、同一训练集增强管线(含苦味加权)下，'
               '对比五类回归算法(281种数据库版本实测，compare_algos.py 可复现)。'
               '线性基线Ridge仅0.435，证明味觉映射具有强非线性；'
               '数据库扩至281种后全量回归发现 GradientBoosting 在整体R2、5折交叉验证'
               '(0.810对0.778)与苦味维度(0.713对0.577)上全面反超原方案的 '
               'HistGradientBoosting，且小数据规模下训练更快，'
               '故按「测试驱动选型」原则切换为本作品基础模型，选型成立。'))
cmp_tbl = [
    [P('算法', 'table_h'), P('整体R2', 'table_h'), P('MAE', 'table_h'),
     P('苦味R2', 'table_h'), P('训练耗时', 'table_h')],
    [P('Ridge(线性基线)', 'table_l'), P('0.435', 'table_c'), P('1.142', 'table_c'),
     P('0.160', 'table_c'), P('<0.1s', 'table_c')],
    [P('RandomForest', 'table_l'), P('0.730', 'table_c'), P('0.707', 'table_c'),
     P('0.564', 'table_c'), P('0.3s', 'table_c')],
    [P('MLP(64,32)', 'table_l'), P('0.593', 'table_c'), P('0.791', 'table_c'),
     P('0.445', 'table_c'), P('1.3s', 'table_c')],
    [P('HistGBR+长尾加权(原方案)', 'table_l'), P('0.758', 'table_c'),
     P('0.633', 'table_c'), P('0.577', 'table_c'), P('3.5s', 'table_c')],
    [P('GradientBoosting+长尾加权(本作品)', 'table_l'), P('0.780', 'table_c'),
     P('0.615', 'table_c'), P('0.713', 'table_c'), P('1.7s', 'table_c')],
]
story.append(std_table(cmp_tbl, [4.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.2 * cm]))
story.append(Spacer(1, 4 * mm))
story.append(H('4.4', '典型案例验证'))
case = [
    [P('食品/配方', 'table_h'), P('预测六维味觉(甜/酸/咸/苦/鲜/涩)', 'table_h'),
     P('常识一致性', 'table_h')],
    [P('香蕉', 'table_l'), P('7.80 / 1.80 / 1.06 / 1.52 / 2.39 / 1.78', 'table_l'),
       P('高甜低酸，正确', 'table_c')],
    [P('柠檬', 'table_l'), P('1.89 / 8.84 / 1.04 / 2.82 / 2.24 / 2.26', 'table_l'),
       P('极酸，正确', 'table_c')],
    [P('山楂', 'table_l'), P('4.27 / 6.28 / 1.02 / 1.89 / 1.38 / 4.69', 'table_l'),
       P('酸中带涩，正确', 'table_c')],
    [P('黑咖啡', 'table_l'), P('0.95 / 1.13 / 1.82 / 7.88 / 1.90 / 2.93', 'table_l'),
       P('高苦带涩，正确', 'table_c')],
    [P('绿茶', 'table_l'), P('0.97 / 1.02 / 1.03 / 5.04 / 1.06 / 4.03', 'table_l'),
       P('苦涩并存，正确', 'table_c')],
    [P('酱油', 'table_l'), P('1.16 / 1.79 / 9.17 / 1.76 / 7.52 / 1.05', 'table_l'),
       P('咸鲜突出，正确', 'table_c')],
    [P('西红柿炒蛋(2:1)', 'table_l'),
       P('1.66 / 1.80 / 3.04 / 1.67 / 5.04 / 1.37', 'table_l'),
       P('鲜味主导，健康分72.5', 'table_c')],
]
story.append(std_table(case, [3.0 * cm, 7.5 * cm, 3.5 * cm]))
story.append(Spacer(1, 4 * mm))
story.append(H('4.5', '讨论: 苦味优化的两步走与涩味实验否决'))
story.append(P('酸味维度在引入有机酸后R2由0.639跃升至0.890(13特征时期实测)，验证了「味觉物质直接入模」'
               '这一技术路线的有效性。苦味优化沿同一思路分两步完成: 其一，扩充特征——'
               '苦味物质谱系复杂，除多酚外还有咖啡因(咖啡/茶)、胡椒碱/苦瓜苷等生物碱(香辛料/瓜类)、'
               '单宁(茶/红酒/柿子)，二期新增咖啡因、苦味生物碱、单宁三个特征后，'
               '特征与苦味标签的相关性由多酚0.46提升至生物碱0.57、单宁0.51；'
               '其二，样本加权——苦味标注呈长尾分布(苦味≥5的样本仅占10%)，'
               '对高苦味样本加权后模型不再被低苦味样本主导。两步使苦味R2由0.407提升至0.619；'
               '数据库扩至281种后按测试驱动选型切换 GradientBoosting 基础模型，苦味R2进一步提升至0.713。'
               '剩余误差主要来自苦味肽、异硫氰酸酯等未收录物质，以及人类苦味感知的'
               '个体基因差异(如TAS2R38受体多态性)带来的标注噪声。'))
story.append(P('针对归因分析发现的涩味多因素耦合、R2偏低(0.609)问题，我们复用苦味'
               '长尾加权策略设计了5折交叉验证对照实验: 对涩味>=5样本分别加权2.0/1.5倍'
               '(>=7加权2.5/2.0倍)，结果涩味R2仅提升0.005/0.001，整体R2反而微降。'
               '这印证了归因结论——涩味缺少单一主导物质变量，重加权无法补充缺失信息，'
               '收益为噪声级，故予以否决并保留现方案。该实验同时说明: 涩味的进一步提升'
               '必须依靠数据扩充与针对性特征工程，已列入展望部分。'))
story.append(H('4.6', '真实用户试用验证'))
story.append(P('实验室指标之外，我们招募 22 名试用者(食品相关专业 14 人)完成被试内验证: '
               '每人先凭自身感受为 3 种熟悉食品的六维味觉打分，再查看系统预测并评吻合度。'
               '共获 63 条可比对的「主观打分-模型预测」数据对。'))
uf = [[P('真实世界验证指标', 'table_h'), P('结果', 'table_h'), P('说明', 'table_h')],
      [P('主导味觉一致率', 'table_c'), P('82.5%(52/63)', 'table_c'),
       P('用户感受最强的维度即模型预测最强维度', 'table_l')],
      [P('用户-预测相关系数', 'table_c'), P('0.84(均值)', 'table_c'),
       P('同一食品六维轮廓的皮尔逊相关', 'table_l')],
      [P('预测吻合度主观评分', 'table_c'), P('3.97/5', 'table_c'),
       P('4 分及以上占 71.4%', 'table_l')],
      [P('总体满意度', 'table_c'), P('8.05/10', 'table_c'),
       P('8 分及以上占 61.9%; 有用度 4.0/5', 'table_l')]]
story.append(std_table(uf, [4.2 * cm, 3.0 * cm, 7.5 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(P('主导味觉不一致的 11 条记录中，5 条为复合食品品种/做法差异(均属咸鲜相邻'
               '维度分歧)、3 条为明显随意填写、3 条为个体感知差异，未发现系统性偏差方向。'
               '试用者将「低糖低钠配方研发」选为第一大应用场景(12/22)，与本作品的应用定位'
               '相互印证; 总体满意度 8.05/10。详细数据见《用户试用反馈报告》及附带的 '
               'survey_analysis.json(可复算)。'))
story.append(H('4.7', '外部效度验证: 文献锚点与真实菜谱'))
story.append(P('留出测试与交叉验证属于内部效度，证明模型复现了训练标签的统计规律；'
               '而「模型预测是否符合独立于训练数据的感官科学共识」需要外部效度检验。'
               '为此设计两类独立于训练集的验证(src/external_validation.py 与 '
               'src/recipe_validation.py，随测试套件一键复算)。'))
story.append(P('(a) 文献锚点验证: 选取 32 个有明确文献/标准共识的感官锚点——'
               'A类为 ISO 8586 与 GB/T 12310 感官分析标准规定的五大基本味标准参照物'
               '(蔗糖-甜、氯化钠-咸、柠檬酸-酸、咖啡因-苦、谷氨酸钠-鲜等 6 项)，'
               'B类为食品感官科学教材与风味化学研究中的 26 项公认结论'
               '(海带谷氨酸为鲜味发现之源、香菇鸟苷酸、柿单宁涩感、啤酒花苦味、'
               '菠菜草酸涩感等)——检验模型预测值是否落入文献区间。'
               '实测 32/32 全部命中(100%)，A类标准参照物 6/6 全中，平均越界偏差 0.0。'))
story.append(P('(b) 真实菜谱一致性验证: 选取 20 道烹饪界主导味型有明确共识的经典菜谱'
               '(冰糖炖梨、糖醋排骨、咸菜白粥、香菇炖鸡汤、凉拌苦瓜等)，'
               '按真实用料配比折算为每100g营养成分，经正向模型+风味交互修正层预测六维味觉，'
               '检验主导味觉与烹饪共识的一致性。实测期望味型落入预测前 2 位的比例为 '
               '20/20(100%)，落入第 1 位为 17/20(85%)；未居首位的 3 道'
               '(糖醋排骨、咸蛋瘦肉粥、苦瓜炒蛋)预测首位均为鲜味、期望味型均列第二，'
               '属相邻味型排序差异，无系统性偏差。'))
story.append(P('局限性声明: 锚点区间取自独立于本数据库标签体系的文献共识，'
               '但训练标签标注时同样参考了领域共识，故上述验证检验的是'
               '「模型是否稳定复现领域共识而非过拟合标签噪声」，'
               '属于外部效度的必要非充分证据。'))
story.append(H('4.8', '不确定性量化: 交叉保形预测区间'))
story.append(P('点估计之外，系统为每条预测附加 80% 预测区间，回答「预测值有多可信」。'
               '方法采用交叉保形预测(Cross-Conformal Prediction): 以主模型 5 折交叉的'
               '折外(out-of-fold)预测残差为校准集，按保形分位数 ceil((n+1)q)/n 取'
               '各味觉维度绝对残差分位数作为半宽，区间 = 点估计 ± 半宽。'
               '相比分位数回归，保形预测具备分布无关的有限样本覆盖保证'
               '(校准集与新样本可交换时，覆盖率不低于目标置信水平)，'
               '且区间恒以点估计为中心，不与任何已发布的点预测数字矛盾。'))
uq_tbl = [
    [P('指标', 'table_h'), P('数值', 'table_h')],
    [P('目标置信度 / 校准集规模', 'table_l'), P('80% / 281 条(5 折折外)', 'table_l')],
    [P('实测覆盖率(各维均值)', 'table_l'), P('0.806(分维 0.804~0.808)', 'table_l')],
    [P('各维区间半宽', 'table_l'),
     P('甜 1.038 / 酸 0.788 / 咸 0.823 / 苦 0.749 / 鲜 1.144 / 涩 1.011', 'table_l')],
    [P('平均区间宽度(0-10 量表)', 'table_l'), P('1.851', 'table_l')],
]
story.append(std_table(uq_tbl, [5.4 * cm, 9.3 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(P('应用价值: 区间已集成至 Web 应用(配方分析页与味觉预测页的折叠面板)与 '
               'API(/api/taste、/api/recipe 响应新增 taste_interval_80 字段)。'
               '窄区间样本(如苦 ±0.749)可直接用于配方筛选；宽区间样本提示人工复核，'
               '为低糖低钠配方研发提供「何时可信、何时存疑」的决策依据。'
               '示例: 黑咖啡苦味点估计 7.77，80% 区间 [7.02, 8.52]。'))

# 五 应用
story.append(H('五、', '应用实现'))
story.append(H('5.1', 'Web应用功能模块'))
story.append(P('系统以 Streamlit 构建交互式Web应用，Plotly 绘制雷达图/柱状图，'
               '七大功能模块覆盖从查询到创造的完整链路:'))
mod = [
    [P('模块', 'table_h'), P('功能', 'table_h')],
    [P('首页概览', 'table_l'),
     P('数据库规模、模型指标、各味觉R2与类别分布可视化、技术路线说明', 'table_l')],
    [P('食品查询', 'table_l'),
     P('按类别检索281种食品，展示16项营养与风味化学特征(有机酸/多酚/咖啡因/苦味生物碱/单宁)、'
       '专家味觉雷达、健康评分与搭配建议', 'table_l')],
    [P('配方分析', 'table_l'),
     P('多选食材并设置克数，按用量加权计算混合营养，AI预测整道菜的六维味觉与健康评分',
       'table_l')],
    [P('逆向设计', 'table_l'),
     P('给定目标味觉与低钠/低糖约束，按应用场景反向搜索最优食材组合与克数，'
       '输出Top-3多样化方案并给出预测vs目标雷达对比', 'table_l')],
    [P('味觉预测', 'table_l'),
     P('16个滑块手动输入检测数据(支持一键载入参考食品)，实时输出味觉雷达、风味描述与建议',
       'table_l')],
    [P('营养对比', 'table_l'),
     P('2-6种食品横向对比基础营养、五类风味化学特征与味觉雷达，附明细表', 'table_l')],
    [P('关于项目', 'table_l'), P('技术方案、关键指标、创新点与局限展望', 'table_l')],
]
story.append(std_table(mod, [2.6 * cm, 12.1 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(H('5.2', 'API服务化: 三层架构开放味觉计算能力'))
story.append(P('模型以 FastAPI 封装为标准 RESTful 服务，自带交互式接口文档(Swagger UI)，'
               '整体架构分为三层: 感知层负责多通道数据输入(拍照OCR、手动录入、食品库检索)，'
               '服务层承载模型推理与营养分析，应用层面向最终用户与第三方系统。'
               '味觉计算由此从单一演示系统升级为可被菜谱App、营养管理、食品研发工具'
               '直接集成的开放服务。'))
story.append(Spacer(1, 2 * mm))
story.append(arch_diagram())
story.append(Spacer(1, 3 * mm))
api_tbl = [
    [P('接口', 'table_h'), P('方法', 'table_h'), P('功能', 'table_h')],
    [P('/api/taste', 'table_l'), P('POST', 'table_c'),
     P('输入16项营养化学指标，输出六维味觉预测、80%预测区间、风味描述与搭配建议', 'table_l')],
    [P('/api/analyze', 'table_l'), P('POST', 'table_c'),
     P('附加健康评分、五级健康等级与各营养素每日推荐占比的完整分析', 'table_l')],
    [P('/api/foods /search /{name}', 'table_l'), P('GET', 'table_c'),
     P('食品库检索: 全量列表(可按类别过滤)、关键词搜索、单食品营养与味觉详情', 'table_l')],
    [P('/api/recipe', 'table_l'), P('POST', 'table_c'),
     P('多食材按克数加权混合，预测整道菜/饮品的味觉(含80%预测区间)与健康评分', 'table_l')],
    [P('/api/design', 'table_l'), P('POST', 'table_c'),
     P('逆向配方设计: 目标味觉+约束条件反向搜索食材组合与克数，返回多样化方案', 'table_l')],
    [P('/api/ocr', 'table_l'), P('POST', 'table_c'),
     P('上传营养成分表照片，OCR识别并解析为结构化营养数据', 'table_l')],
]
story.append(std_table(api_tbl, [3.6 * cm, 1.4 * cm, 9.7 * cm]))
story.append(Spacer(1, 3 * mm))
story.append(P('全部接口已在真实调用中验证通过。例如向 /api/recipe 提交苦瓜150g+苹果200g，'
               '返回混合营养与预测味觉(甜5.8/苦3.5)，并给出"甜味适中，带有苦味，风味平衡"'
               '的描述，与常识一致；黑咖啡经 /api/taste 预测苦味为主导味，符合实际。'))
story.append(H('5.3', '拍照识别: 营养成分表OCR输入通道'))
story.append(P('为进一步降低未知食品的使用门槛，系统在味觉预测页集成拍照识别通道: '
               '用户对准食品包装上的营养成分表拍照上传，RapidOCR(基于ONNX的本地推理引擎，'
               '支持中文、无需联网与外部API)提取文本后，规则解析模块自动定位能量、蛋白质、'
               '脂肪、碳水化合物、钠等国家强制标注项——能量自动完成千焦到千卡的换算，'
               '解析过程排除反式脂肪、碳酸氢钠等干扰词并进行合理范围校验，'
               '识别结果一键填入16项输入滑块完成预测。在程序生成的标准营养成分表'
               '端到端测试中，能量、蛋白质、脂肪、碳水化合物、钠五项强制标注字段'
               '识别全部正确。'))
story.append(P('识别鲁棒性经专项测试迭代加固: 端到端测试曾发现"钠"等单字字段名'
               '易被OCR检测器丢弃、且含NRV%列的检测框顺序交错导致字段-数值失配，'
               '团队据此实施三项修复——其一，按检测框纵坐标聚类还原表格行、行内按横坐标'
               '排序后再解析；其二，确认目标为营养表但钠字段缺失时自动放大2倍重识别；'
               '其三，依据GB 28050强制标识顺序(能量/蛋白质/脂肪/碳水化合物/钠)，'
               '当核心四项齐全且仅剩一个未被其他字段消费的毫克值时将其归属钠字段。'
               '修复后含NRV%列标签的五项字段端到端识别全部通过，该案例也体现了'
               '测试驱动的工程闭环。'))
story.append(H('5.4', '部署方案与测试验证'))
story.append(P('系统为纯CPU应用，无需GPU与外部服务，模型在服务启动时于内存中即时训练'
               '(约7秒)，支持三种部署方式: 本地 pip install -r requirements.txt 后'
               '以 streamlit run src/app.py 启动Web应用、以 uvicorn 启动API服务；'
               '或通过随附 Dockerfile 构建单容器镜像，同时暴露Web(8501)与API(8001)双端口。'
               '系统已永久部署于 Streamlit Community Cloud, 访问地址 '
               'https://nutritongue.streamlit.app, 无需安装即可体验全部功能'
               '(免费版闲置休眠, 首次访问约30-60秒唤醒; 云端OCR自动降级为手动输入)。'
               '项目提供可一键复现的专项测试套件 src/run_tests.py，覆盖模型性能、'
               '边界规则、典型食品常识一致性、预测延迟(1.6ms)、OCR端到端解析、API接口、'
               '文献锚点/真实菜谱外部效度、不确定性量化与演示数字冻结守卫'
               '共43项用例，当前全部通过(详见随附《测试报告》)。'))
story.append(P('技术栈: Python 3.13 · pandas · NumPy · scikit-learn · Streamlit · Plotly · '
               'FastAPI · RapidOCR · ReportLab。项目包含数据库CSV、核心算法模块、API服务、'
               'Web应用、本技术方案与requirements清单，'
               '可通过 pip install -r requirements.txt 后 streamlit run app.py 一键复现。',
               'body_ni'))

# 六 创新点
story.append(H('5.5', '应用场景与价值'))
story.append(P('本作品的核心价值在于: 在所有「不能先尝、尝不起、尝不过来」的环节替代或预筛感官评价。'
               '主场景为低糖/低钠配方研发——在「三减三健」与健康中国2030政策推动下，'
               '食品企业普遍面临减盐减糖后的口感维持难题，而传统研发依赖专业感官评价小组，'
               '单次评价需组织培训评价员、费用数千元、周期数天，配方迭代成本极高。'
               '味析可在配方设计阶段对候选配方秒级输出六维味觉预测，'
               '把「改一版、尝一次」变为「筛百版、尝三版」，'
               '让真人感官评价只集中于最优的少数方案，显著降低研发成本与周期。'))
story.append(P('延伸场景包括: 预制菜口味一致性质控(原料批次波动→理化指标波动→预测味觉偏差→'
               '指导配方微调)；慢病人群膳食依从管理(低钠饮食最大障碍是适口性差，'
               '拍照识别营养表加味觉预测帮助患者「既达标、又吃得下」)；'
               '食品专业教学(以归因热力图直观呈现「化学组成→感官表现」的映射规律)。'
               '国际上 Gastrograph AI 等公司已验证味觉预测的商业模式，'
               '本作品以零硬件、低成本的方式实现了同类能力的校园级复现与开放化。'))
story.append(Spacer(1, 3 * mm))
story.append(H('六、', '创新点'))
for t in [
    '从「尝味道」到「设计味道」的逆向配方设计: 以训练后的味觉模型为评估器，'
    '按场景约束食材池随机搜索数万组食材组合与配比，经目标距离与低钠/低糖/油脂/'
    '调味品占比罚分排序、批量局部精修与多样性挑选，约3秒输出匹配度90%左右的'
    '多样化配方方案，实现从感知到创造的能力闭环；',
    '零硬件电子舌范式: 不使用任何物理传感器与实验样本，以数据驱动方式完成电子舌核心功能，'
    '把硬件门槛降为零，便于在校园场景中复制与推广；',
    '味觉化学先验入模: 在营养特征集中引入有机酸、多酚、咖啡因、苦味生物碱、单宁五个'
    '物理可测量的风味物质变量，酸味R2由0.639提升至0.844、苦味R2由0.407提升至0.713，'
    '特征选择与味觉生理机制严格对应，科学可解释；',
    '从单品走向配方: 支持任意食材、任意配比的混合营养加权计算与混合味觉推演，'
    '可用于菜品设计、健康饮食指导，而非停留在单食品查询；',
    '机理与数据双重可解释: 排列重要性归因显示每个味觉维度的主导特征恰为其化学本质'
    '(甜←糖分1.34、咸←钠1.51、酸←有机酸1.04、鲜←蛋白质1.02、'
    '苦←生物碱/单宁/咖啡因)，证明模型学到的是风味化学规律而非统计噪声；',
    '可信赖预测: 交叉保形预测为每条输出附带具备有限样本覆盖保证的80%预测区间，'
    '配合32项文献锚点、20道真实菜谱两重外部效度验证与演示数字冻结守卫测试，'
    '把「精度指标」升级为「何时可信、何时存疑」的可决策信息；',
    '服务化与多模态输入: 模型封装为FastAPI标准接口对外开放，支持营养成分表拍照OCR识别，'
    '未知食品拍一下就能尝，从演示系统升级为可被集成的味觉计算服务；',
    '严谨的实验设计: 类别分层留出 + 5折交叉验证 + 训练集定向数据增强，'
    '并对超参数、随机森林等替代模型做过对比实验，结论稳健。',
]:
    story.append(Paragraph('• ' + t, ss['bullet']))

# 七 展望
story.append(H('七、', '局限与展望'))
for t in [
    '特征扩充: 引入苦味肽、氨基酸(谷氨酸等)、核苷酸、嘌呤、有机酸细分谱、'
    '异硫氰酸酯等，进一步提升苦味等长尾味觉的预测精度；',
    '数据扩充: 将数据库从281种扩展至千种以上，覆盖加工食品与地方风味，并引入多人标注'
    '降低主观噪声；',
    '模型升级: 尝试神经网络多任务学习、特征对数变换与交互项，并针对归因分析'
    '发现的涩味多因素耦合问题做专门建模；',
    '个性化校准: 增加个体味觉敏感度问卷，实现千人千味的个性化味觉映射；',
    '场景延伸: 对接膳食记录、菜谱推荐与慢病低钠饮食管理，形成完整的AI营养助手。',
]:
    story.append(Paragraph('• ' + t, ss['bullet']))

# 八 参考文献
story.append(H('八、', '参考文献'))
for t in [
    '[1] GB 28050-2011 食品安全国家标准 预包装食品营养标签通则. 北京: 中国标准出版社, 2011.',
    '[2] 中国营养学会. 中国居民膳食指南(2022). 北京: 人民卫生出版社, 2022.',
    '[3] 杨月欣. 中国食物成分表(标准版·第6版). 北京: 北京大学医学出版社, 2019.',
    '[4] Friedman J H. Greedy function approximation: a gradient boosting machine. '
    'Annals of Statistics, 2001, 29(5): 1189-1232.',
    '[5] Pedregosa F, et al. Scikit-learn: machine learning in Python. '
    'Journal of Machine Learning Research, 2011, 12: 2825-2830.',
    '[6] Ke G, et al. LightGBM: a highly efficient gradient boosting decision tree. '
    'NeurIPS, 2017.',
    '[7] 电子舌技术在食品味觉评价中的应用研究进展. 食品科学, 2021.',
    '[8] Keast R S J, Breslin P A S. An overview of binary taste-taste interactions. '
    'Food Quality and Preference, 2003, 14(2): 111-124.',
    '[9] RapidOCR: 基于ONNXRuntime的跨平台中文OCR开源库. '
    'https://github.com/RapidAI/RapidOCR, 2024.',
]:
    story.append(Paragraph(t, ss['bullet']))

# 附录A 核心代码
story.append(H('附录A', '核心代码片段'))
story.append(P('以下为本作品最关键的三段算法实现(完整代码见随附源码包 src/ 目录)。',
               'body_ni'))
story.append(P('A.1 模型构建与苦味长尾样本加权(taste_model.py):', 'h2'))
for line in [
    "def _make_base(self):",
    "    # 281种数据库上经五类算法对比选定(留出R2与5折CV均最优)",
    "    return GradientBoostingRegressor(random_state=42)",
    "",
    "sample_weight = np.ones(len(y_aug))",
    "sample_weight[y_aug[:, bitter_idx] >= 5] = 2.5",
    "sample_weight[y_aug[:, bitter_idx] >= 7] = 3.0",
]:
    story.append(P(line.replace(' ', '&nbsp;') if line else '&nbsp;', 'code'))
story.append(Spacer(1, 2 * mm))
story.append(P('A.2 零输入边界规则(taste_model.py):', 'h2'))
for line in [
    "if all(float(nutrition_dict.get(c, 0)) <= 0 for c in self.TASTE_BEARING):",
    "    return {col: 0.0 for col in self.target_cols}",
]:
    story.append(P(line.replace(' ', '&nbsp;'), 'code'))
story.append(Spacer(1, 2 * mm))
story.append(P('A.3 OCR 孤立毫克值归属(nutrition_ocr.py, GB 28050 顺序先验):', 'h2'))
for line in [
    "core = ('calories', 'protein', 'fat', 'carbs')",
    "if 'sodium' not in parsed and all(k in parsed for k in core):",
    "    orphans = [float(v) for v in mg_vals if round(float(v), 2) not in used]",
    "    if len(orphans) == 1 and lo <= orphans[0] <= hi:",
    "        parsed['sodium'] = round(orphans[0], 2)",
]:
    story.append(P(line.replace(' ', '&nbsp;'), 'code'))

# ================================================================ 附录B 软件设计资料
story.append(H('附录B', '软件设计资料'))


def _fbox(d, x, y, w, h, lines, fill, fs=8.0, bold=False):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=HexColor('#d1d5db'),
               strokeWidth=0.8, rx=3, ry=3))
    n = len(lines)
    for i, t in enumerate(lines):
        yy = y + h / 2 + (n - 1) * 5.5 - i * 11 - 2.5 if n > 1 else y + h / 2 - 3
        d.add(String(x + w / 2, yy, t, fontName=FONT_B if bold else FONT,
                     fontSize=fs, textAnchor='middle', fillColor=DARK))


def arch_diagram():
    d = Drawing(425, 205)
    layers = [
        ('用户层', ['浏览器(Web交互界面)', '第三方系统(HTTP API)', '用户试用问卷'],
         162, 30, HexColor('#fdf2f8')),
        ('应用层', ['app.py · Streamlit 七大功能页', 'api_server.py · FastAPI 六个端点'],
         122, 30, LIGHT),
        ('算法层', ['taste_model 预测/交互修正/评分', 'inverse_design 逆向设计 · nutrition_ocr 拍照识别',
                  'feature_importance 可解释性归因'],
         72, 40, HexColor('#eff6ff')),
        ('数据层', ['food_database.csv · 281 种食品 × 25 字段',
                  '(16 维输入特征 + 6 维味觉标签 + 元数据)'],
         24, 30, HexColor('#f0fdf4')),
    ]
    for name, lines, y, h, color in layers:
        d.add(String(50, y + h / 2 - 3, name, fontName=FONT_B, fontSize=9.5,
                     textAnchor='end', fillColor=ORANGE))
        _fbox(d, 58, y, 360, h, lines, color, fs=8.0)
    for y in (152, 112, 54):
        d.add(Line(240, y, 240, y + 10, strokeColor=GRAY, strokeWidth=1.2))
    d.add(String(212, 195, '系统总体架构图(四层)', fontName=FONT_B, fontSize=10.5,
                 textAnchor='middle', fillColor=DARK))
    return d


def flow_diagram(title, steps, note):
    d = Drawing(425, 88)
    d.add(String(212, 76, title, fontName=FONT_B, fontSize=10.5,
                 textAnchor='middle', fillColor=DARK))
    bw, gap, y = 64, 5, 34
    for i, s in enumerate(steps):
        x = 6 + i * (bw + gap)
        _fbox(d, x, y, bw, 26, [s], LIGHT, fs=7.5)
        if i < len(steps) - 1:
            d.add(Line(x + bw, y + 13, x + bw + gap, y + 13,
                       strokeColor=ORANGE, strokeWidth=1.4))
    d.add(String(212, 14, note, fontName=FONT, fontSize=8.5,
                 textAnchor='middle', fillColor=GRAY))
    return d


story.append(P('B.1 系统架构', 'h2'))
story.append(arch_diagram())
story.append(Spacer(1, 2 * mm))
story.append(P('B.2 模块设计(关键设计参数)', 'h2'))
mod_rows = [['模块', '源文件', '职责', '关键设计参数']]
for r in [
    ['味觉预测核心', 'taste_model.py', '16 特征到 6 维味觉的多输出回归;风味交互修正;健康与综合评分',
     'GradientBoosting(200 树/lr 0.05/subsample 0.8);苦味样本加权 2.5/3.0;高斯噪声增强 x4'],
    ['逆向设计', 'inverse_design.py', '目标味觉到配方的约束优化搜索',
     '2.4 万组采样(3 种组合数 x 8000)+Top120 精修(30 轮);原料白名单(排除 39 种预包装成品);添加糖占比<=20%;随机种子固定'],
    ['OCR 识别', 'nutrition_ocr.py', '营养成分表拍照识别为结构化营养字典',
     'RapidOCR(ONNX 推理);GB 28050 强制标识 1+4 顺序先验;千焦到千卡换算'],
    ['可解释性', 'feature_importance.py', '16 特征 x 6 味觉排列重要性归因', 'n_repeats=30'],
    ['Web 应用', 'app.py', '首页/查询/配方/逆向/预测/对比/关于 七大功能页', 'Streamlit;启动时一次性训练并缓存'],
    ['API 服务', 'api_server.py', '/api/taste、/api/recipe、/api/design 等六个端点', 'FastAPI + uvicorn;自动文档 /docs'],
]:
    mod_rows.append([Paragraph(c, ss['table_c']) if i != 3 else Paragraph(c, ss['table_l'])
                     for i, c in enumerate(r)])
story.append(std_table(mod_rows, [1.7 * cm, 2.5 * cm, 5.2 * cm, 7.6 * cm]))
story.append(Spacer(1, 2 * mm))
story.append(P('B.3 核心流程', 'h2'))
story.append(flow_diagram(
    '正向味觉预测流程',
    ['输入 16 维特征', '边界规则校验', 'GB 多输出回归', '风味交互修正',
     '健康/综合评分', '六维结果输出'],
    '交互修正含甜苦拮抗、咸鲜协同、甜酸互抑、蛋白-单宁沉淀四条文献规则;综合评分=风味 x0.6+健康 x0.4-搭配扣减'))
story.append(Spacer(1, 2 * mm))
story.append(flow_diagram(
    '逆向配方设计流程',
    ['目标味觉+场景', '原料白名单池', '采样组合配比', '批量预测评分',
     '约束罚分排序', '精修输出 Top3'],
    '约束罚分含低钠/低糖国标限值、添加糖与油脂占比上限、类别一致性;全程以正向模型为评估器,无生成式幻觉'))

# ================================================================ 附录C 测试数据
story.append(H('附录C', '测试数据'))
story.append(P('C.1 测试环境: Windows 11 / Python 3.13 / scikit-learn 1.x / '
               '测试脚本 src/run_tests.py 一键执行,共 43 项断言式用例。', 'body_ni'))
story.append(P('C.2 测试用例执行结果(原始记录):', 'h2'))
test_rows = [['编号', '用例(期望)', '实测结果', '结论']]
for r in [
    ['M-01', '留出测试整体 R2>=0.70', '0.780', 'PASS'],
    ['M-02', '5 折交叉验证整体 R2>=0.70', '0.8096', 'PASS'],
    ['M-03', 'MAE<=0.70', '0.6152', 'PASS'],
    ['M-04', '各味觉 R2 无低于 0.5 的失效维度', '最低=0.609(涩味)', 'PASS'],
    ['B-01', '全零输入(白水)判定无味', '六维输出全 0.0', 'PASS'],
    ['B-02', '仅含糖 5g 时不误判无味', '甜味 5.65', 'PASS'],
    ['S-06', '绿茶主导味觉=苦味', '主导=苦味, MAE=0.04', 'PASS'],
    ['S-07', '黑咖啡主导味觉=苦味', '主导=苦味, MAE=0.12', 'PASS'],
    ['S-08', '苹果主导味觉=甜味', '主导=甜味, MAE=0.22', 'PASS'],
    ['S-09', '食盐主导味觉=咸味', '主导=咸味, MAE=0.10', 'PASS'],
    ['P-01', '单次味觉预测延迟', '1.6ms', 'PASS'],
    ['D-01', '饮品场景食材池合规且匹配度>=85%', '90.7%(苹果汁+哈密瓜+白糖)', 'PASS'],
    ['D-02', '低钠约束下钠含量<=125mg/100g', '108mg/100g', 'PASS'],
    ['D-03', '纯油脂类合计占比<=30%', '0%', 'PASS'],
    ['D-04', '同参数两次调用结果一致(种子固定)', '完全一致', 'PASS'],
    ['D-05', '单次逆向设计耗时<10s', '0.6s', 'PASS'],
    ['I-01', '苦瓜+白糖修正后苦味下降(甜苦拮抗)', '3.45 -> 2.22', 'PASS'],
    ['I-02', '高咸+鲜配方修正后鲜味不降(咸鲜协同)', '7.24 -> 8.54', 'PASS'],
    ['I-03', '奶茶配方修正后涩味下降(蛋白-单宁沉淀)', '1.86 -> 1.84', 'PASS'],
    ['G-01~05', '演示案例点预测冻结守卫(黑咖啡/牛奶/苦瓜+苹果/白粥+三文鱼)',
     '与已发布材料数字逐值一致', 'PASS'],
    ['E-01~03', '文献锚点外部效度(32锚点命中率>=75%, A类ISO/GB标准参照物全中)',
     '命中32/32=100%, A类6/6, 平均越界偏差0.0', 'PASS'],
    ['R-01~02', '真实菜谱一致性(20道top2>=80%, top1>=70%)',
     'top2=20/20=100%, top1=17/20=85%', 'PASS'],
    ['U-01~03', '不确定性量化(覆盖率0.75~0.90, 半宽<=1.5, 区间含点估计)',
     '覆盖率0.806, 最大半宽1.144, 苦7.77[7.02,8.52]', 'PASS'],
    ['O-xx', 'OCR 营养表五字段识别(能量/蛋白质/脂肪/碳水/钠)',
     '358.51kcal、5.2g、8.1g、20.3g、450mg(5 项全对)', 'PASS'],
    ['A-01~06', 'API 六个端点(/、taste、search、recipe、foods、design)',
     '全部 HTTP 200,design 匹配度 90.7%', 'PASS'],
]:
    test_rows.append([r[0], Paragraph(r[1], ss['table_l']),
                      Paragraph(r[2], ss['table_l']), r[3]])
story.append(std_table(test_rows, [1.5 * cm, 7.8 * cm, 6.0 * cm, 1.5 * cm]))
story.append(Spacer(1, 2 * mm))
story.append(P('C.3 性能指标数据:', 'h2'))
perf_rows = [['指标', '数值', '统计口径'],
             ['留出测试 R2', '0.780', '分层抽样 20%(57 条测试集)'],
             ['5 折交叉验证 R2', '0.810', 'KFold(shuffle, seed=42)'],
             ['MAE / RMSE', '0.615 / 0.920', '六维平均'],
             ['分维 R2', '甜 0.860 / 酸 0.844 / 咸 0.883 / 苦 0.713 / 鲜 0.772 / 涩 0.609', '留出测试集'],
             ['单次预测延迟', '1.6ms', '本机实测均值'],
             ['逆向设计耗时', '约 0.6s', '2 万组采样+精修全程'],
             ['模型训练耗时', '10.3s', '含数据增强与交叉验证'],
             ['测试通过率', '43/43(100%)', 'run_tests.py 一键复算']]
perf_rows = [[Paragraph(c, ss['table_l']) if j == 2 else c for j, c in enumerate(row)]
             for row in perf_rows]
story.append(std_table(perf_rows, [4.2 * cm, 7.2 * cm, 5.4 * cm]))
story.append(Spacer(1, 2 * mm))
story.append(P('C.4 数据简要分析:', 'h2'))
story.append(P('模型可信度方面,交叉验证 R2(0.810)高于留出测试(0.780),说明模型未过拟合;'
               'S 系列样本主导味觉全部命中且 MAE<0.3,验证了对典型食品的判别能力。'
               '外部效度方面,32 个文献锚点全部落入文献区间、20 道真实菜谱 top2 '
               '一致率 100%,证明模型稳定复现领域感官共识而非过拟合标签噪声;'
               '交叉保形区间实测覆盖率 0.806 与名义 0.80 吻合,不确定性量化可信。'
               '边界健壮性方面,全零输入正确返回无味、OCR 孤立毫克值按 GB 28050 顺序正确归属钠字段。'
               '工程可用性方面,预测亚毫秒、逆向设计亚秒、API 六端点全通,满足在线交互要求。'
               '全部用例可经 run_tests.py 一键复算,原始执行日志与性能数据如上。'))

# ================================================================ 附录D 佐证材料
story.append(H('附录D', '佐证材料清单'))
story.append(P('1. 用户使用证明: 《用户试用反馈报告》——22 名试用者(含 14 名食品专业)完成被试内测试,'
               '主导味觉一致率 82.5%,用户-预测六维相关系数 0.84,总体满意度 8.05/10;'
               '随附 22 份问卷原始统计数据(JSON)。详见提交材料 05-用户反馈/ 目录。', 'body_ni'))
story.append(P('2. 检测佐证: 本方案附录 C 所列 43 项测试用例执行记录,'
               '以及《测试报告》中 15 条改进记录(含实验对照与否决实验),'
               '均可经源码包内 run_tests.py、compare_algos.py 一键复算验证。', 'body_ni'))
story.append(P('3. 说明: 本作品为在校学生竞赛作品,暂无专利证书与第三方检测报告。', 'body_ni'))

story.append(Spacer(1, 8 * mm))
story.append(HRFlowable(width='100%', thickness=0.8, color=ORANGE))
story.append(Spacer(1, 3 * mm))
story.append(P('味析 NutriTongue —— 让AI像人一样尝味道，用数据为健康饮食决策服务。',
               'subtitle'))

# ---------------------------------------------------------------- 构建
class TplDoc(BaseDocTemplate):
    """把 H() 登记的一/二级标题写入目录"""

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and hasattr(fl, '_toc'):
            lvl, txt = fl._toc
            self.notify('TOCEntry', (lvl, txt, self.page))


# 官方模板页面设置: 上下2.5cm 左右3cm, 页眉页脚1.5cm, A4纵向
doc = TplDoc(str(OUT), pagesize=A4,
             leftMargin=3 * cm, rightMargin=3 * cm,
             topMargin=2.5 * cm, bottomMargin=2.5 * cm,
             title='味析NutriTongue技术方案', author='味来科技')
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id='normal')
doc.addPageTemplates([PageTemplate(id='main', frames=frame, onPage=on_page)])
doc.multiBuild(story)
print('PDF generated:', OUT)
