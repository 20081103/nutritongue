# -*- coding: utf-8 -*-
"""
味析 NutriTongue - AI虚拟电子舌
Streamlit 交互应用：食品查询 / 配方分析 / 味觉预测 / 营养对比
2026第八届全球校园人工智能算法精英大赛 - 赛题2: AI+软件创新
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import (  # noqa: E402
    TastePredictionModel, NutritionAnalyzer, FoodEvaluator,
    apply_taste_interactions,
)
import inverse_design as _inverse_design  # noqa: E402
# Streamlit 重跑(Rerun)会复用进程内已导入的模块; 仓库源码更新后若进程未重启,
# 会出现"新 app.py + 旧 inverse_design"的错配导致的 TypeError, 这里强制重载
importlib.reload(_inverse_design)  # noqa: E402
from inverse_design import design_recipes  # noqa: E402
from uncertainty import TasteUncertainty  # noqa: E402

# ---------------------------------------------------------------- 路径与常量
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'

TASTE_ORDER = ['sweet', 'sour', 'salty', 'bitter', 'umami', 'astringent']
TASTE_CN = {'sweet': '甜味', 'sour': '酸味', 'salty': '咸味',
            'bitter': '苦味', 'umami': '鲜味', 'astringent': '涩味'}
TASTE_COLORS = {'sweet': '#ec4899', 'sour': '#eab308', 'salty': '#3b82f6',
                'bitter': '#92400e', 'umami': '#ef4444', 'astringent': '#8b5cf6'}

FEATURE_CN = {
    'calories': '热量(kcal)', 'protein': '蛋白质(g)', 'carbs': '碳水化合物(g)',
    'fat': '脂肪(g)', 'fiber': '膳食纤维(g)', 'sugar': '糖分(g)',
    'sodium': '钠(mg)', 'potassium': '钾(mg)', 'calcium': '钙(mg)',
    'iron': '铁(mg)', 'vitamin_c': '维生素C(mg)',
    'organic_acid': '有机酸(g)', 'polyphenol': '多酚(mg)',
    'caffeine': '咖啡因(mg)', 'alkaloid': '苦味生物碱(mg)', 'tannin': '单宁(mg)'
}
FEATURE_HELP = {
    'organic_acid': '柠檬酸/苹果酸/酒石酸等有机酸总含量，是酸味的物质基础',
    'polyphenol': '茶多酚/花青素/黄酮等多酚总含量，与苦味、涩味密切相关',
    'caffeine': '咖啡因等甲基黄嘌呤，咖啡/茶/巧克力特有苦味来源',
    'alkaloid': '胡椒碱/苦瓜苷/花椒酰胺等非咖啡因苦味生物碱',
    'tannin': '缩合与水解单宁，茶/红酒/未熟柿子的苦涩味来源'
}

st.set_page_config(
    page_title='味析 NutriTongue - AI虚拟电子舌',
    page_icon='👅', layout='wide',
    initial_sidebar_state='expanded'
)

CUSTOM_CSS = """
<style>
    .main-header {font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg,#f97316,#ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-header {color: #6b7280; font-size: 1.05rem; margin-top: -8px;}
    .metric-card {background: #fff; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); border-left: 5px solid #f97316;
        color: #1f2937;}
    .metric-value {font-size: 1.9rem; font-weight: 800; color: #111827;}
    .metric-label {color: #6b7280; font-size: .9rem;}
    .info-box {background: #fff7ed; border-radius: 12px; padding: 16px 20px;
        border-left: 5px solid #f97316; margin: 10px 0; color: #1f2937;}
    .score-ring {text-align:center; padding:10px; color: #1f2937;}
    .taste-tag {display:inline-block; padding:3px 12px; border-radius:20px;
        color:#fff; font-size:.85rem; margin:2px;}
    section[data-testid="stSidebar"] {background: #fff7ed;}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {color: #1f2937;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- 缓存加载
@st.cache_resource(show_spinner='正在加载281种食品数据库并训练味觉模型...')
def load_engine():
    df = pd.read_csv(DATA_PATH)
    model = TastePredictionModel()
    metrics = model.train(str(DATA_PATH))
    return df, model, metrics


DF, MODEL, METRICS = load_engine()
ANALYZER = NutritionAnalyzer()
EVALUATOR = FoodEvaluator()
# 交叉保形预测区间(80%置信): 出厂校准半宽秒级装载, 为点预测附加可信度
UQ = TasteUncertainty()


def interval_panel(pred):
    """预测区间展示面板(加法组件, 不改变任何既有展示)"""
    iv = UQ.interval_for(pred)
    with st.expander('📏 预测区间（80%置信 · 交叉保形预测）', expanded=False):
        tbl = pd.DataFrame([{
            '维度': TASTE_CN[k], '点预测': f'{pred[k]:.1f}',
            '80%预测区间': f'{iv[k][0]:.1f} ~ {iv[k][1]:.1f}'}
            for k in TASTE_ORDER])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption('区间越窄代表该维度预测越可信；由主模型5折交叉折外残差'
                   '保形校准，覆盖率实测80.6%，区间恒以点预测为中心。')


# ---------------------------------------------------------------- 绘图工具
def taste_radar(taste_dict, title='味觉雷达图', fill_color='rgba(249,115,22,0.35)'):
    """绘制六维味觉雷达图，taste_dict 的键为英文味觉名"""
    labels = [TASTE_CN[k] for k in TASTE_ORDER]
    values = [float(taste_dict.get(k, 0)) for k in TASTE_ORDER]
    fig = go.Figure(data=go.Scatterpolar(
        r=values + values[:1], theta=labels + labels[:1],
        fill='toself', fillcolor=fill_color,
        line=dict(color='#f97316', width=2.5), name=title
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], tickfont_size=11)),
        margin=dict(l=40, r=40, t=50, b=20), height=380,
        title=dict(text=title, x=0.5, font_size=15)
    )
    return fig


def dual_taste_radar(taste_a, name_a, taste_b, name_b, title='味觉对比'):
    labels = [TASTE_CN[k] for k in TASTE_ORDER]
    va = [float(taste_a.get(k, 0)) for k in TASTE_ORDER]
    vb = [float(taste_b.get(k, 0)) for k in TASTE_ORDER]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=va + va[:1], theta=labels + labels[:1],
                                  fill='toself', name=name_a,
                                  fillcolor='rgba(249,115,22,0.25)',
                                  line=dict(color='#f97316')))
    fig.add_trace(go.Scatterpolar(r=vb + vb[:1], theta=labels + labels[:1],
                                  fill='toself', name=name_b,
                                  fillcolor='rgba(59,130,246,0.20)',
                                  line=dict(color='#3b82f6')))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        margin=dict(l=40, r=40, t=50, b=20), height=400,
        title=dict(text=title, x=0.5, font_size=15)
    )
    return fig


def nutrient_bars(analysis, title='每日推荐摄入量占比'):
    rows = [(v['name'], v['pct']) for v in analysis.values()]
    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    pcts = [min(r[1], 200) for r in rows]
    colors = ['#ef4444' if p > 100 else '#f97316' if p > 60 else '#22c55e'
              for p in pcts]
    fig = go.Figure(go.Bar(x=pcts, y=names, orientation='h',
                           marker_color=colors, text=[f'{p:.0f}%' for p in pcts],
                           textposition='outside'))
    fig.update_layout(title=title, margin=dict(l=10, r=40, t=40, b=10),
                      height=380, xaxis_title='占每日推荐量百分比(%)',
                      yaxis=dict(autorange='reversed'))
    return fig


def taste_tags(taste_dict):
    parts = []
    for k in TASTE_ORDER:
        v = taste_dict.get(k, 0)
        parts.append(
            f'<span class="taste-tag" style="background:{TASTE_COLORS[k]}">'
            f'{TASTE_CN[k]} {v:.1f}</span>')
    return '<div>' + ''.join(parts) + '</div>'


def score_block(score, label='健康评分'):
    level, advice, color = ANALYZER.get_health_level(score)
    st.markdown(
        f'<div class="score-ring"><div style="font-size:3rem;font-weight:800;'
        f'color:{color}">{score}</div>'
        f'<div style="font-size:1.1rem;font-weight:600">{level}</div>'
        f'<div style="color:#6b7280">{advice}</div></div>',
        unsafe_allow_html=True)


def row_nutrition(row):
    return {c: float(row[c]) for c in MODEL.feature_cols}


# ---------------------------------------------------------------- 侧边栏
st.sidebar.markdown('## 👅 味析 NutriTongue')
st.sidebar.caption('AI 虚拟电子舌 · 尝得到味道，算得清营养')
PAGES = ['🏠 首页概览', '🔍 食品查询', '🍲 配方分析',
         '🧭 逆向设计', '🧪 味觉预测', '📊 营养对比', 'ℹ️ 关于项目']
PAGE_SLUGS = {'home': 0, 'query': 1, 'recipe': 2, 'design': 3,
              'predict': 4, 'compare': 5, 'about': 6}
# 支持通过 URL 查询参数直达页面，如 ?page=recipe
_init_idx = PAGE_SLUGS.get(st.query_params.get('page', 'home'), 0)
page = st.sidebar.radio('功能导航', PAGES, index=_init_idx)
st.sidebar.markdown('---')
st.sidebar.markdown(
    f'**数据库**：{METRICS["n_samples"]} 种食品 · 9 大类<br>'
    f'**输入特征**：16 维营养/化学指标<br>'
    f'**输出味觉**：6 维味觉强度(0-10)',
    unsafe_allow_html=True)
st.sidebar.caption('2026第八届全球校园人工智能算法精英大赛 · 赛题2 AI+软件创新')


# ================================================================= 首页
if page.startswith('🏠'):
    st.markdown('<div class="main-header">味析 · NutriTongue</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 虚拟电子舌：用营养数据"尝"出味道，'
                '用机器学习理解风味</div>', unsafe_allow_html=True)

    st.markdown('### 🎯 项目简介')
    st.markdown(
        '<div class="info-box">传统电子舌依赖昂贵的物理传感器阵列。本项目反其道而行之：'
        '以 <b>16 维可测量的营养与食品化学指标</b>（11 项基础营养 + '
        '<b>有机酸</b>、<b>多酚</b>、<b>咖啡因</b>、<b>苦味生物碱</b>与<b>单宁</b>五类风味物质）为输入，'
        '通过多输出梯度提升回归模型，'
        '预测食品在 <b>甜、酸、咸、苦、鲜、涩</b> 六个维度上的味觉强度，'
        '并结合膳食指南给出营养评价、配方分析与风味搭配建议。'
        '纯软件方案，零硬件成本，覆盖 9 大类 281 种常见食品。</div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (f'{METRICS["n_samples"]} 种', '食品样本量'),
        ('9 大类', '食品类别覆盖'),
        ('16 维', '输入特征'),
        ('6 维', '味觉输出'),
        (f'{METRICS["test_r2"]:.3f}', '留出测试 R²'),
    ]
    for col, (val, lab) in zip([c1, c2, c3, c4, c5], cards):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}'
                     f'</div><div class="metric-label">{lab}</div></div>',
                     unsafe_allow_html=True)

    st.markdown('### 📈 模型性能')
    cc1, cc2 = st.columns(2)
    with cc1:
        r2_df = pd.DataFrame({
            '味觉维度': list(METRICS['per_taste_r2'].keys()),
            'R2': list(METRICS['per_taste_r2'].values())
        })
        fig = go.Figure(go.Bar(
            x=r2_df['味觉维度'], y=r2_df['R2'],
            marker_color=['#ec4899', '#eab308', '#3b82f6', '#92400e',
                          '#ef4444', '#8b5cf6'],
            text=r2_df['R2'].round(3), textposition='outside'))
        fig.update_layout(title='各味觉维度预测决定系数 R²(留出测试集)',
                          yaxis=dict(range=[0, 1], title='R²'),
                          margin=dict(t=50, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        cat_counts = DF['category'].value_counts().reset_index()
        cat_counts.columns = ['类别', '数量']
        fig2 = go.Figure(go.Bar(
            x=cat_counts['数量'], y=cat_counts['类别'], orientation='h',
            marker_color='#f97316', text=cat_counts['数量'], textposition='outside'))
        fig2.update_layout(title='数据库类别分布', margin=dict(t=50, b=10, r=40),
                           height=360, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('### ✨ 核心亮点')
    h1, h2, h3 = st.columns(3)
    h1.info('**🧪 化学可解释性**\n\n引入有机酸、多酚两个物理可测量特征，'
            '酸味预测 R² 由 0.64 提升至 0.89，建模逻辑与味觉生理机制一致。')
    h2.success('**🤖 多任务机器学习**\n\n多输出梯度提升回归一次预测 6 种味觉；'
               '高斯噪声数据增强 + 分层留出 + 5 折交叉验证，泛化可靠。')
    h3.warning('**🍲 实用化场景**\n\n支持单食品查询、混合配方味觉模拟、'
               '自定义营养指标试算与多食品营养对比，贴近真实饮食决策。')

    st.markdown('### 🔬 工作流程')
    st.markdown(
        '**营养/化学数据输入** → 标准化缩放 → 多输出 GBDT 集成模型 → '
        '六维味觉强度 → 雷达图可视化 + 风味描述 → 营养健康评分 + 搭配建议')
    m1, m2, m3 = st.columns(3)
    m1.metric('5折交叉验证 R²', f'{METRICS["cv_r2"]:.4f}')
    m2.metric('RMSE 均方根误差', f'{METRICS["test_rmse"]:.3f}')
    m3.metric('MAE 平均绝对误差', f'{METRICS["test_mae"]:.3f}')


# ================================================================= 食品查询
elif page.startswith('🔍'):
    st.title('🔍 食品查询')
    st.caption('选择任意食品，查看其营养构成、专家标注味觉与健康评价')

    cat = st.selectbox('按类别筛选', ['全部'] + sorted(DF['category'].unique()))
    pool = DF if cat == '全部' else DF[DF['category'] == cat]
    food_name = st.selectbox('选择食品', pool['name'].tolist())
    row = pool[pool['name'] == food_name].iloc[0]

    st.markdown(f'### {food_name}  <small style="color:#f97316">'
                f'{row["category"]}</small>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        nutr = row_nutrition(row)
        analysis = ANALYZER.analyze(nutr, servings=100)
        st.plotly_chart(nutrient_bars(analysis), use_container_width=True)

        st.markdown('#### 每 100g 营养成分明细')
        detail = pd.DataFrame([
            {'指标': FEATURE_CN[c], '含量': f'{nutr[c]:g}'}
            for c in MODEL.feature_cols if c in ANALYZER.DAILY_REF
        ])
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.markdown('#### ⚗️ 风味化学特征')
        st.markdown(
            f'- **有机酸**：{nutr["organic_acid"]:g} g/100g '
            f'<span style="color:#9ca3af">（酸味物质基础）</span><br>'
            f'- **多酚**：{nutr["polyphenol"]:g} mg/100g '
            f'<span style="color:#9ca3af">（苦味/涩味相关物质）</span><br>'
            f'- **咖啡因**：{nutr["caffeine"]:g} mg/100g '
            f'<span style="color:#9ca3af">（咖啡/茶特有苦味碱）</span><br>'
            f'- **苦味生物碱**：{nutr["alkaloid"]:g} mg/100g '
            f'<span style="color:#9ca3af">（胡椒碱/苦瓜苷等）</span><br>'
            f'- **单宁**：{nutr["tannin"]:g} mg/100g '
            f'<span style="color:#9ca3af">（茶/红酒苦涩味来源）</span>',
            unsafe_allow_html=True)
    with right:
        actual = {k: float(row[k]) for k in TASTE_ORDER}
        st.plotly_chart(taste_radar(actual, f'{food_name} · 专家味觉标注'),
                        use_container_width=True)
        st.markdown(taste_tags(actual), unsafe_allow_html=True)
        hs = ANALYZER.health_score(nutr)
        score_block(hs)
        overall = EVALUATOR.overall_score(actual, hs)
        st.metric('综合风味评分', f'{overall} / 100')
        st.markdown(f'**风味描述**：{EVALUATOR.taste_description(actual)}')
        st.markdown(f'**搭配建议**：{EVALUATOR.pairing_suggestion(actual, row["category"])}')


# ================================================================= 配方分析
elif page.startswith('🍲'):
    st.title('🍲 配方分析')
    st.caption('选择多种食材并设置用量，模拟整道菜/混合物的营养与味觉表现')

    # 容错：过滤掉旧会话中可能失效的食材名（如数据库更名）
    valid_names = set(DF['name'])
    _default_items = [('西红柿', 100), ('鸡蛋', 50)]
    _items = [(n, g) for n, g in st.session_state.get('recipe_items', _default_items)
              if n in valid_names]
    st.session_state.recipe_items = _items or _default_items

    with st.form('recipe_form'):
        st.markdown('#### 添加食材（可多选）')
        chosen = st.multiselect(
            '食材', DF['name'].tolist(),
            default=[n for n, _ in st.session_state.recipe_items],
            key='recipe_choice')
        st.markdown('#### 各项用量（克）')
        cols = st.columns(3)
        grams = {}
        prev_map = dict(st.session_state.recipe_items)
        cat_map = dict(zip(DF['name'], DF['category']))
        for i, name in enumerate(chosen):
            with cols[i % 3]:
                is_cond = cat_map.get(name) == '调味品'
                # 调味品用量小、对配比敏感，给更细刻度与更小默认值
                grams[name] = st.number_input(
                    f'{name} (g)', min_value=1.0, max_value=2000.0,
                    value=float(prev_map.get(name, 10 if is_cond else 100)),
                    step=0.5 if is_cond else 5.0, format='%.1f',
                    key=f'g_{name}')
        submitted = st.form_submit_button('🧮 开始分析', type='primary')

    if submitted and chosen:
        st.session_state.recipe_items = [(n, grams[n]) for n in chosen]
        total_g = sum(grams.values())
        # 各成分含量均以每100g计，按用量加权平均即得混合物每100g含量
        mix_per100 = pd.Series(0.0, index=MODEL.feature_cols)
        for name in chosen:
            r = DF[DF['name'] == name].iloc[0]
            w = grams[name] / total_g
            for c in MODEL.feature_cols:
                mix_per100[c] += float(r[c]) * w
        mix_dict = {c: float(mix_per100[c]) for c in MODEL.feature_cols}

        st.markdown('---')
        st.success(f'配方总重量 **{total_g:g} g**，以下营养与味觉均按 '
                   f'**每 100g 混合食材** 计（模拟烹饪前混合状态）')

        k1, k2 = st.columns([3, 2])
        with k1:
            st.markdown('#### 混合营养构成（每100g）')
            mix_tbl = pd.DataFrame([
                {'指标': FEATURE_CN[c], '每100g含量': round(mix_dict[c], 2)}
                for c in MODEL.feature_cols
            ])
            st.dataframe(mix_tbl, use_container_width=True, hide_index=True)
            st.plotly_chart(
                nutrient_bars(ANALYZER.analyze(mix_dict), '各营养素占每日推荐量比例'),
                use_container_width=True)
        with k2:
            pred_raw = MODEL.predict(mix_dict)
            pred, inter_notes = apply_taste_interactions(pred_raw, mix_dict)
            st.plotly_chart(taste_radar(pred, 'AI 预测配方味觉',
                                        fill_color='rgba(236,72,153,0.30)'),
                            use_container_width=True)
            st.markdown(taste_tags(pred), unsafe_allow_html=True)
            interval_panel(pred)
            if inter_notes:
                st.caption('风味交互修正已生效：' + '；'.join(inter_notes))
            hs = ANALYZER.health_score(mix_dict)
            score_block(hs, label='配方健康评分')

        st.markdown('#### 📋 配方评价')
        st.markdown(f'**风味描述**：{EVALUATOR.taste_description(pred)}')
        st.markdown(f'**搭配建议**：{EVALUATOR.pairing_suggestion(pred)}')
        # ---- 搭配合理性规则层：模型输出为六维化学味觉预测，"好不好吃/搭不搭"
        # ---- 属于风味和谐度问题，由烹饪常识规则层单独提示并扣分
        PUNGENT = {'大蒜', '生姜', '辣椒', '洋葱', '韭菜', '小葱', '花椒', '孜然',
                   '黑胡椒', '白胡椒', '辣椒粉', '咖喱粉', '姜黄粉', '芥末酱'}
        OFFAL = {'猪血', '猪肝', '鸡肝'}          # 血制品/内脏：铁+蛋白含量高
        STRONG_ACID = {'柠檬', '山楂', '食醋', '柠檬水'}  # 实测酸度≥6的强酸食材
        RAW_DAIRY = {'牛奶', '奶油', '奶粉'}      # 未发酵乳：酪蛋白遇酸变性凝结
        sweet_map = dict(zip(DF['name'], DF['sweet']))
        penalty = 0
        conflicts = []

        pungent_in = [n for n in chosen if n in PUNGENT]
        dessert_in = [n for n in chosen
                      if n not in PUNGENT and sweet_map.get(n, 0) >= 6]
        if pungent_in and dessert_in:
            penalty += 15
            conflicts.append(
                f'{"、".join(pungent_in)} 属刺激性香辛料，'
                f'与 {"、".join(dessert_in)}(高甜) 的组合在烹饪中极少见')

        offal_in = [n for n in chosen if n in OFFAL]
        fruit_in = [n for n in chosen if cat_map.get(n) == '水果']
        if offal_in and fruit_in:
            penalty += 15
            conflicts.append(
                f'{"、".join(offal_in)} 为血制品/内脏，富含铁与蛋白质；'
                f'{"、".join(fruit_in)} 等水果的有机酸与单宁会与之结合，'
                f'产生腥涩异味，传统烹饪从不与水果同烹')

        acid_in = [n for n in chosen if n in STRONG_ACID]
        dairy_in = [n for n in chosen if n in RAW_DAIRY]
        if acid_in and dairy_in:
            penalty += 10
            conflicts.append(
                f'{"、".join(acid_in)} 为强酸食材，会使 {"、".join(dairy_in)} '
                f'中的酪蛋白变性凝结、口感分层结块(酸凝奶酪原理)，直接混合食用体验差')

        for c in conflicts:
            st.warning(f'⚠️ 搭配合理性提示：{c}。AI 输出为化学味觉强度预测，'
                       f'不代表风味和谐度；综合评分已相应扣减。')
        cond_share = sum(grams[n] for n in chosen
                         if cat_map.get(n) == '调味品') / total_g
        if cond_share > 0.2:
            st.info(f'ℹ️ 调味品合计占比 {cond_share:.0%}，高于常见菜肴水平(约5-15%)，'
                    f'预测结果可能偏咸/偏重口。')
        bd = EVALUATOR.overall_breakdown(pred, hs)
        overall = max(0, round(bd['overall'] - penalty, 1))
        st.metric('配方综合评分', f'{overall} / 100')
        cap = (f"评分构成：风味 {bd['flavor_score']} ×60% = {bd['flavor_weighted']}"
               f" ＋ 健康 {bd['health_score']} ×40% = {bd['health_weighted']}")
        if penalty:
            cap += f" － 搭配扣减 {penalty}"
        st.caption(cap)
    elif not chosen:
        st.info('请至少选择一种食材')


# ================================================================= 逆向设计
elif page.startswith('🧭'):
    st.title('🧭 逆向配方设计')
    st.caption('给定想要的味道，AI 在食材库中反向搜索最优组合与用量'
               '——从"AI 尝味道"到"AI 设计味道"')

    TGT_DEFAULT = {'sweet': 8.0, 'sour': 2.0, 'salty': 0.0,
                   'bitter': 0.0, 'umami': 0.0, 'astringent': 0.0}
    with st.form('design_form'):
        st.markdown('#### 目标味觉强度（0-10）')
        tc = st.columns(6)
        target = {}
        for i, c in enumerate(TASTE_ORDER):
            with tc[i]:
                target[c] = st.slider(TASTE_CN[c], 0.0, 10.0,
                                      TGT_DEFAULT[c], 0.5, key=f'tgt_{c}')
        o1, o2, o3, o4, o5 = st.columns(5)
        with o1:
            scenario = st.selectbox('应用场景', ['通用', '饮品·甜品', '菜肴·主食'])
        with o2:
            total_g = st.number_input('总用量（克）', 100, 1000, 300, 50)
        with o3:
            low_sodium = st.checkbox('低钠（≤120mg/100g）')
        with o4:
            low_sugar = st.checkbox('低糖（≤5g/100g）')
        with o5:
            mode = st.selectbox('输出类型', ['食品设计', '饮食搭配'],
                                help='食品设计=单一成品(如一道菜/一杯饮); '
                                     '饮食搭配=一餐组合(各食材可独立成菜一起食用)')
        submitted = st.form_submit_button('🎯 开始设计', type='primary')

    if submitted:
        if all(v <= 0 for v in target.values()):
            st.warning('请至少为一个味觉维度设置非零目标')
        else:
            try:
                with st.spinner('正在搜索数万种「食材组合 × 配比」并精修...'):
                    mode_arg = 'meal' if mode == '饮食搭配' else 'product'
                    _res = design_recipes(MODEL, DF, target,
                                          total_grams=int(total_g), scenario=scenario,
                                          low_sodium=low_sodium, low_sugar=low_sugar,
                                          mode=mode_arg, n_results=3)
                st.session_state['design_results'] = (dict(target), scenario, mode, _res)
            except Exception as _e:
                import traceback as _tb
                st.error('设计过程出现异常，请调整参数后重试。')
                st.code(f'{type(_e).__name__}: {_e}\n\n{_tb.format_exc()}')

    if 'design_results' in st.session_state:
        _tgt, _sc, _mode, _res = st.session_state['design_results']
        st.markdown('---')
        st.markdown(f'### 设计方案（{_sc} · {_mode}）')
        cols = st.columns(3)
        for i, r in enumerate(_res):
            with cols[i]:
                items_html = '<br>'.join(
                    f"· {it['name']} <b>{it['grams']}g</b>" for it in r['items'])
                if r['style'] == '饮食搭配':
                    style_tag = '�️ 饮食搭配'
                elif r['style'] == '需烹饪食谱':
                    style_tag = '🍳 需烹饪食谱'
                else:
                    style_tag = '🥗 生食/即食搭配'
                st.markdown(
                    f'<div class="metric-card"><b>方案 {i+1}</b> · 匹配度 '
                    f'<span style="color:#f97316;font-weight:800">'
                    f'{r["match_score"]}%</span> · <span style="color:#2563eb">'
                    f'{style_tag}</span><br>{items_html}</div>',
                    unsafe_allow_html=True)
                nu = r['nutrition_per100g']
                st.caption(f"每100g：{nu['calories']:.0f}kcal · "
                           f"糖{nu['sugar']:.1f}g · 钠{nu['sodium']:.0f}mg")
                st.plotly_chart(
                    dual_taste_radar(r['predicted_taste'], 'AI预测',
                                     _tgt, '目标', f'方案{i+1}：预测 vs 目标'),
                    use_container_width=True, key=f'design_radar_{i}')
        st.markdown(
            '<div class="info-box">算法说明：按场景约束食材池后随机采样数万组'
            '「食材组合 × 配比」，以训练好的味觉模型批量评估，'
            '经低钠/低糖/调味品占比罚分排序、配比局部精修与多样性挑选输出 '
            'Top-3 方案；随机种子固定，结果可复现。</div>',
            unsafe_allow_html=True)
        st.caption(
            '说明：味觉预测基于原料营养按克数加和计算，未建模烹饪过程'
            '（美拉德反应、焦糖化、蛋白质变性等）对风味的影响。'
            '「🍽️ 饮食搭配」为一餐内可独立成菜的食材组合，按混合营养评估整体味型；'
            '「🥗 生食/即食搭配」预测可信度较高；'
            '「🍳 需烹饪食谱」的预测反映原料混合的基础味型，'
            '实际成品味道可能因烹饪方式产生差异，建议作为食材选型与配比参考。')

# ================================================================= 味觉预测
elif page.startswith('🧪'):
    st.title('🧪 味觉预测实验室')
    st.caption('手动输入 16 项营养与化学指标（如检测报告数据），AI 实时预测味觉')

    preset = st.selectbox('快速载入参考食品（再手动微调）', ['不载入'] + DF['name'].tolist())
    base_row = None
    if preset != '不载入':
        base_row = DF[DF['name'] == preset].iloc[0]

    st.markdown('#### 输入指标（每 100g 含量）')
    inputs = {}
    grid = st.columns(3)
    ranges = {
        'calories': (0.0, 900.0, 1.0), 'protein': (0.0, 50.0, 0.1),
        'carbs': (0.0, 100.0, 0.1), 'fat': (0.0, 100.0, 0.1),
        'fiber': (0.0, 60.0, 0.1), 'sugar': (0.0, 100.0, 0.1),
        'sodium': (0.0, 40000.0, 10.0), 'potassium': (0.0, 6000.0, 10.0),
        'calcium': (0.0, 1000.0, 1.0), 'iron': (0.0, 60.0, 0.1),
        'vitamin_c': (0.0, 200.0, 0.1),
        'organic_acid': (0.0, 6.0, 0.05), 'polyphenol': (0.0, 3000.0, 10.0),
        'caffeine': (0.0, 100.0, 0.5), 'alkaloid': (0.0, 300.0, 1.0),
        'tannin': (0.0, 500.0, 1.0)
    }

    with st.expander('📷 拍照识别营养成分表（自动填入滑块）'):
        st.caption('对准食品包装的"营养成分表"拍摄，OCR 自动提取数据'
                   '（按每100g计，能量自动将千焦换算为千卡）')
        up = st.file_uploader('上传营养成分表照片',
                              type=['jpg', 'jpeg', 'png', 'webp'])
        if up is not None:
            try:
                from nutrition_ocr import parse_from_image
                data = up.getvalue()
                if st.session_state.get('ocr_hash') != hash(data):
                    with st.spinner('OCR 识别中…'):
                        _, parsed = parse_from_image(data)
                    st.session_state.ocr_hash = hash(data)
                    st.session_state.ocr_parsed = parsed
                parsed = st.session_state.get('ocr_parsed', {})
                if parsed:
                    show = pd.DataFrame([
                        {'指标': FEATURE_CN.get(k, k), '识别值(每100g)': v}
                        for k, v in parsed.items()])
                    st.dataframe(show, hide_index=True, use_container_width=True)
                    if st.button('✏️ 填入下方滑块'):
                        for k, v in parsed.items():
                            if k in ranges:
                                lo, hi, _ = ranges[k]
                                st.session_state[f'slide_{k}'] = float(
                                    min(max(v, lo), hi))
                        st.success('已填入，可在下方微调')
                else:
                    st.warning('未识别到有效营养数据，'
                               '请更清晰地对准营养成分表区域拍摄')
            except ImportError:
                st.info('当前部署环境未启用 OCR 组件，请手动输入下方指标')
            except RuntimeError as e:
                st.warning(str(e))

    # 关键：预设食品改变时，必须在滑块组件创建前写入 session_state，
    # 否则带 key 的滑块会保留旧值，value 参数不生效
    if st.session_state.get('last_preset') != preset:
        st.session_state.last_preset = preset
        if base_row is not None:
            for c in MODEL.feature_cols:
                lo, hi, _ = ranges[c]
                st.session_state[f'slide_{c}'] = float(
                    min(max(float(base_row[c]), lo), hi))

    for i, c in enumerate(MODEL.feature_cols):
        lo, hi, step = ranges[c]
        default = float(base_row[c]) if base_row is not None else 0.0
        default = min(max(default, lo), hi)
        with grid[i % 3]:
            inputs[c] = st.slider(
                FEATURE_CN[c], min_value=float(lo), max_value=float(hi),
                value=st.session_state.get(f'slide_{c}', float(default)),
                step=float(step),
                help=FEATURE_HELP.get(c, None), key=f'slide_{c}')

    # 营养标签(GB 28050)不强制标糖: 糖分为0而碳水较高时, 按碳水50%估算糖分,
    # 否则甜饮/糕点会被误判为不甜; 主食等不甜食品可取消勾选
    if inputs.get('sugar', 0) <= 0 and inputs.get('carbs', 0) > 10:
        est = round(inputs['carbs'] * 0.5, 1)
        if st.checkbox(
                f'🔢 糖分为 0 而碳水 {inputs["carbs"]:.1f}g：营养标签通常不单独'
                f'标糖，已按碳水的 50% 估算糖分 {est}g'
                f'（若为米饭/面条/馒头等主食，请取消勾选）',
                value=True):
            inputs['sugar'] = est

    pred = MODEL.predict(inputs)
    st.markdown('---')
    p1, p2 = st.columns([3, 2])
    with p1:
        st.plotly_chart(taste_radar(pred, 'AI 预测味觉结果',
                                    fill_color='rgba(236,72,153,0.30)'),
                        use_container_width=True)
    with p2:
        st.markdown('#### 六维味觉强度')
        st.markdown(taste_tags(pred), unsafe_allow_html=True)
        interval_panel(pred)
        for k in TASTE_ORDER:
            st.slider(TASTE_CN[k], 0.0, 10.0, float(pred[k]), 0.1,
                      disabled=True)
        st.markdown(f'**风味描述**：{EVALUATOR.taste_description(pred)}')
        st.markdown(f'**搭配建议**：{EVALUATOR.pairing_suggestion(pred)}')

    hs = ANALYZER.health_score(inputs)
    st.markdown('#### 营养健康评价')
    q1, q2, q3 = st.columns([1, 1, 2])
    with q1:
        score_block(hs)
    with q2:
        st.metric('综合风味评分',
                  f'{EVALUATOR.overall_score(pred, hs)} / 100')
    with q3:
        st.plotly_chart(nutrient_bars(ANALYZER.analyze(inputs),
                                      '营养素占每日推荐量比例'),
                        use_container_width=True)


# ================================================================= 营养对比
elif page.startswith('📊'):
    st.title('📊 营养对比')
    st.caption('选择 2-6 种食品，横向对比营养与味觉差异')
    foods = st.multiselect('选择对比食品', DF['name'].tolist(),
                           default=['苹果', '香蕉', '牛肉'])
    if len(foods) < 2:
        st.info('请至少选择两种食品')
    else:
        sub = DF[DF['name'].isin(foods)].set_index('name').loc[foods]

        tab1, tab2, tab3 = st.tabs(['基础营养对比', '风味物质对比', '味觉雷达对比'])
        with tab1:
            metric = st.selectbox(
                '选择营养指标',
                [c for c in MODEL.feature_cols if c in ANALYZER.DAILY_REF],
                format_func=lambda x: FEATURE_CN[x])
            fig = go.Figure(go.Bar(
                x=sub.index, y=sub[metric], marker_color='#f97316',
                text=sub[metric].round(1), textposition='outside'))
            fig.update_layout(title=f'{FEATURE_CN[metric]} 对比（每100g）',
                              yaxis_title=FEATURE_CN[metric],
                              margin=dict(t=50, b=10), height=400)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            flavor_chems = [
                ('organic_acid', '#eab308'), ('polyphenol', '#8b5cf6'),
                ('caffeine', '#92400e'), ('alkaloid', '#ef4444'),
                ('tannin', '#6b7280'),
            ]
            for i in range(0, len(flavor_chems), 2):
                pair = flavor_chems[i:i+2]
                cols = st.columns(len(pair))
                for j, (metric, color) in enumerate(pair):
                    with cols[j]:
                        fig = go.Figure(go.Bar(
                            x=sub.index, y=sub[metric],
                            marker_color=color,
                            text=sub[metric].round(2), textposition='outside'))
                        fig.update_layout(title=f'{FEATURE_CN[metric]} 对比（每100g）',
                                          margin=dict(t=50, b=10), height=340)
                        st.plotly_chart(fig, use_container_width=True)
            st.caption('💡 有机酸驱动酸味感知；多酚、咖啡因、苦味生物碱与单宁共同决定苦味与涩感')
        with tab3:
            fig = go.Figure()
            palette = ['#f97316', '#3b82f6', '#22c55e', '#ec4899', '#8b5cf6', '#92400e']
            labels = [TASTE_CN[k] for k in TASTE_ORDER]
            for i, name in enumerate(sub.index):
                vals = [float(sub.loc[name, k]) for k in TASTE_ORDER]
                fig.add_trace(go.Scatterpolar(
                    r=vals + vals[:1], theta=labels + labels[:1],
                    name=name, line=dict(color=palette[i % len(palette)])))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                margin=dict(l=40, r=40, t=50, b=20), height=480,
                title=dict(text='专家标注味觉对比', x=0.5))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('#### 明细数据表')
        show = sub.reset_index()[['name', 'category'] + MODEL.feature_cols + TASTE_ORDER]
        show.columns = ['食品', '类别'] + [FEATURE_CN[c] for c in MODEL.feature_cols] \
            + [TASTE_CN[c] for c in TASTE_ORDER]
        st.dataframe(show, use_container_width=True, hide_index=True)


# ================================================================= 关于
else:
    st.title('ℹ️ 关于项目')
    st.markdown('### 味析 NutriTongue —— AI 虚拟电子舌')
    st.markdown(
        '<div class="info-box">本项目参加 <b>2026 第八届全球校园人工智能算法精英大赛'
        ' · 算法创新赛 · 赛题 2（AI+软件创新）</b>。在不依赖任何物理传感器与实验条件的前提下，'
        '以数据驱动方式模拟电子舌功能，实现"输入营养成分 → 输出味觉特征"的智能推断。</div>',
        unsafe_allow_html=True)

    st.markdown('### 技术方案')
    st.markdown(
        '- **数据层**：自建 281 种食品、9 大类别的结构化数据库，'
        '每条样本含 16 维输入（热量、三大产能营养素、矿物质、维生素、'
        '有机酸、多酚、咖啡因、苦味生物碱、单宁）'
        '与 6 维 0-10 分专家味觉标注。\n'
        '- **算法层**：`MultiOutputRegressor + GradientBoostingRegressor` '
        '多输出梯度提升回归；训练集施加 3 轮高斯乘性噪声增强鲁棒性；'
        '高苦味样本加权以缓解长尾分布；'
        '按食品类别分层留出 20% 测试，并辅以 5 折交叉验证。\n'
        '- **应用层**：Streamlit 构建七大交互模块，Plotly 雷达/柱状图可视化，'
        '配方模块支持加权混合营养计算与混合味觉模拟。\n'
        '- **评价层**：参考《中国居民膳食指南》每日推荐摄入量实现营养画像与百分制健康评分，'
        '并生成风味描述与搭配建议。')

    st.markdown('### 关键指标')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('样本量', f'{METRICS["n_samples"]} 种')
    m2.metric('留出测试 R²', f'{METRICS["test_r2"]:.4f}')
    m3.metric('5折交叉验证 R²', f'{METRICS["cv_r2"]:.4f}')
    m4.metric('MAE', f'{METRICS["test_mae"]:.3f}')

    st.markdown('### 创新点')
    st.markdown(
        '1. **零硬件的电子舌思路**：将味觉感知问题转化为营养化学指标到味觉空间的回归问题。\n'
        '2. **味觉化学先验入模**：新增有机酸、多酚提升酸味 R² 至 0.860；'
        '进一步引入咖啡因、苦味生物碱、单宁三类苦味特征，'
        '苦味 R² 由 0.407 提升至 0.619，科学可解释。\n'
        '3. **从单品到配方**：支持任意食材配比的混合营养与味觉推演，具备实际饮食指导价值。')

    st.markdown('### 局限与展望')
    st.caption(
        '苦味预测(R²≈0.62)已大幅改善但仍为六维中最低，'
        '与苦味肽、异硫氰酸酯等物质未完全收录及个体差异有关；'
        '后续可引入氨基酸、嘌呤、挥发物等特征，扩充数据至千级规模，'
        '并尝试神经网络多任务学习与个性化味觉校准。')

    st.markdown('### 技术栈')
    st.markdown('Python 3.13 · pandas · NumPy · scikit-learn · Streamlit · Plotly · ReportLab')
