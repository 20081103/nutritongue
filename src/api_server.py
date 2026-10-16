# -*- coding: utf-8 -*-
"""
味析 NutriTongue - 味觉计算 API 服务
将 AI 味觉预测模型封装为 RESTful 接口，供菜谱App、营养管理系统、
食品研发工具等第三方系统集成调用。

启动: uvicorn api_server:app --host 0.0.0.0 --port 8000
在线接口文档: http://localhost:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from taste_model import (  # noqa: E402
    TastePredictionModel, NutritionAnalyzer, FoodEvaluator,
    apply_taste_interactions,
)
from inverse_design import design_recipes, SCENARIOS  # noqa: E402
from uncertainty import TasteUncertainty  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'food_database.csv'

MODEL = TastePredictionModel()
ANALYZER = NutritionAnalyzer()
EVALUATOR = FoodEvaluator()
UQ = TasteUncertainty()  # 出厂校准半宽, 装载秒级, 为点预测附加80%预测区间
DF = None


@asynccontextmanager
async def lifespan(app):
    """启动时训练模型并加载数据库（一次性开销，约数秒）"""
    global DF
    metrics = MODEL.train(str(DATA_PATH))
    DF = pd.read_csv(DATA_PATH)
    print(f'[NutriTongue API] 模型已就绪: n={metrics["n_samples"]}, '
          f'test_R2={metrics["test_r2"]}, 苦味R2={metrics["per_taste_r2"].get("苦味")}')
    yield


app = FastAPI(
    title='味析 NutriTongue API',
    version='1.1.0',
    description='AI 虚拟电子舌：从营养成分数据预测六维味觉特征（甜/酸/咸/苦/鲜/涩）'
                '并给出80%保形预测区间，提供营养分析、食品查询、配方混合'
                '与营养成分表OCR识别能力。',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_methods=['*'], allow_headers=['*'],
)


# ------------------------------------------------------------------ 请求模型
class NutritionInput(BaseModel):
    """16项营养与风味化学指标（每100g含量，缺省为0）"""
    calories: float = Field(0, description='热量 kcal')
    protein: float = Field(0, description='蛋白质 g')
    carbs: float = Field(0, description='碳水化合物 g')
    fat: float = Field(0, description='脂肪 g')
    fiber: float = Field(0, description='膳食纤维 g')
    sugar: float = Field(0, description='糖分 g')
    sodium: float = Field(0, description='钠 mg')
    potassium: float = Field(0, description='钾 mg')
    calcium: float = Field(0, description='钙 mg')
    iron: float = Field(0, description='铁 mg')
    vitamin_c: float = Field(0, description='维生素C mg')
    organic_acid: float = Field(0, description='有机酸 g（柠檬酸/苹果酸等）')
    polyphenol: float = Field(0, description='多酚 mg（茶多酚/花青素等）')
    caffeine: float = Field(0, description='咖啡因 mg')
    alkaloid: float = Field(0, description='苦味生物碱 mg（非咖啡因）')
    tannin: float = Field(0, description='单宁 mg')


class RecipeItem(BaseModel):
    food_name: str = Field(None, description='数据库中的食品名称（与nutrition二选一）')
    nutrition: NutritionInput = Field(None, description='自定义营养数据（与food_name二选一）')
    grams: float = Field(100, gt=0, description='用量 g')


class RecipeInput(BaseModel):
    items: list[RecipeItem]


class DesignInput(BaseModel):
    """逆向配方设计请求"""
    target: dict = Field(..., description='目标味觉强度, 键: sweet/sour/salty/'
                                          'bitter/umami/astringent, 值0-10')
    total_grams: int = Field(300, ge=100, le=2000, description='总用量 g')
    k: str = Field('auto', description='每方案食材数: auto 或 2/3/4')
    scenario: str = Field('通用', description='应用场景: 通用/饮品·甜品/菜肴·主食')
    low_sodium: bool = Field(False, description='低钠约束(<=120mg/100g)')
    low_sugar: bool = Field(False, description='低糖约束(<=5g/100g)')
    mode: str = Field('product', description='输出类型: product=食品设计(单一成品), '
                                             'meal=饮食搭配(一餐组合, 放宽类别一致性)')
    n_results: int = Field(3, ge=1, le=5, description='返回方案数')


# ------------------------------------------------------------------ 基础接口
@app.get('/')
def index():
    return {
        'service': '味析 NutriTongue API',
        'version': '1.0.0',
        'docs': '/docs',
        'endpoints': ['/api/taste', '/api/analyze', '/api/foods',
                      '/api/foods/search', '/api/foods/{name}',
                      '/api/recipe', '/api/design', '/api/ocr'],
    }


@app.post('/api/taste')
def predict_taste(n: NutritionInput):
    """核心能力：从营养成分预测六维味觉"""
    pred = MODEL.predict(n.model_dump())
    return {
        'taste': pred,
        'taste_interval_80': UQ.interval_for(pred),
        'description': EVALUATOR.taste_description(pred),
        'pairing': EVALUATOR.pairing_suggestion(pred),
    }


@app.post('/api/analyze')
def full_analyze(n: NutritionInput):
    """完整分析：味觉预测 + 健康评分 + 每日营养占比"""
    d = n.model_dump()
    pred = MODEL.predict(d)
    hs = ANALYZER.health_score(d)
    level, advice, _ = ANALYZER.get_health_level(hs)
    return {
        'taste': pred,
        'description': EVALUATOR.taste_description(pred),
        'pairing': EVALUATOR.pairing_suggestion(pred),
        'health_score': hs,
        'health_level': level,
        'health_advice': advice,
        'overall_score': EVALUATOR.overall_score(pred, hs),
        'daily_pct': ANALYZER.analyze(d),
    }


# ------------------------------------------------------------------ 食品库
@app.get('/api/foods')
def list_foods(category: str = None):
    """列出数据库全部食品（可按类别过滤）"""
    df = DF if category is None else DF[DF['category'] == category]
    return {
        'count': len(df),
        'foods': df[['food_id', 'name', 'category']].to_dict('records'),
    }


@app.get('/api/foods/search')
def search_foods(q: str):
    """按名称关键词搜索食品"""
    hits = DF[DF['name'].str.contains(q, na=False)]
    if hits.empty:
        raise HTTPException(404, f'未找到包含“{q}”的食品')
    return {
        'count': len(hits),
        'foods': hits[['food_id', 'name', 'category']].to_dict('records'),
    }


@app.get('/api/foods/{name}')
def food_detail(name: str):
    """查询食品的营养数据与实测味觉标签"""
    rows = DF[DF['name'] == name]
    if rows.empty:
        raise HTTPException(404, f'数据库中不存在食品“{name}”，'
                                 f'可改用 POST /api/taste 传入营养数据预测')
    r = rows.iloc[0]
    return {
        'name': r['name'],
        'category': r['category'],
        'nutrition': {c: float(r[c]) for c in MODEL.feature_cols},
        'taste': {c: float(r[c]) for c in MODEL.target_cols},
    }


# ------------------------------------------------------------------ 配方
@app.post('/api/recipe')
def analyze_recipe(r: RecipeInput):
    """配方分析：按用量加权混合多种食材，预测整道菜/饮品的味觉"""
    if not r.items:
        raise HTTPException(400, '配方至少包含一种食材')
    mix = {c: 0.0 for c in MODEL.feature_cols}
    total = 0.0
    resolved = []
    for item in r.items:
        if item.food_name:
            rows = DF[DF['name'] == item.food_name]
            if rows.empty:
                raise HTTPException(404, f'数据库中不存在食材“{item.food_name}”')
            nut = {c: float(rows.iloc[0][c]) for c in MODEL.feature_cols}
            label = item.food_name
        elif item.nutrition:
            nut = item.nutrition.model_dump()
            label = '自定义食材'
        else:
            raise HTTPException(400, '每种食材需提供 food_name 或 nutrition')
        for c in MODEL.feature_cols:
            mix[c] += nut[c] * item.grams
        total += item.grams
        resolved.append({'name': label, 'grams': item.grams})
    mix = {c: round(v / total, 3) for c, v in mix.items()}

    pred_raw = MODEL.predict(mix)
    pred, inter_notes = apply_taste_interactions(pred_raw, mix)
    hs = ANALYZER.health_score(mix)
    return {
        'items': resolved,
        'total_grams': round(total, 1),
        'mixed_nutrition_per_100g': mix,
        'taste': pred,
        'taste_interval_80': UQ.interval_for(pred),
        'taste_raw': pred_raw,
        'interactions': inter_notes,
        'description': EVALUATOR.taste_description(pred),
        'pairing': EVALUATOR.pairing_suggestion(pred),
        'health_score': hs,
        'overall_score': EVALUATOR.overall_score(pred, hs),
        'score_breakdown': EVALUATOR.overall_breakdown(pred, hs),
    }


# ------------------------------------------------------------------ 逆向设计
@app.post('/api/design')
def inverse_design_api(inp: DesignInput):
    """逆向配方设计: 目标味觉 -> 搜索最优食材组合与克数"""
    if inp.scenario not in SCENARIOS:
        raise HTTPException(400, f'未知场景, 可选: {list(SCENARIOS)}')
    if all(float(v) <= 0 for v in inp.target.values()):
        raise HTTPException(400, '目标味觉不能全为0')
    results = design_recipes(MODEL, DF, inp.target,
                             total_grams=inp.total_grams, k=inp.k,
                             scenario=inp.scenario,
                             low_sodium=inp.low_sodium, low_sugar=inp.low_sugar,
                             mode=inp.mode, n_results=inp.n_results)
    return {'target': inp.target, 'scenario': inp.scenario, 'mode': inp.mode,
            'total_grams': inp.total_grams, 'results': results}


# ------------------------------------------------------------------ OCR
@app.post('/api/survey')
async def submit_survey(data: dict):
    """收集问卷数据，追加写入 CSV"""
    import csv as csvmod
    from datetime import datetime
    survey_dir = Path(__file__).resolve().parent.parent / 'docs'
    survey_dir.mkdir(exist_ok=True)
    csv_path = survey_dir / 'survey_results.csv'
    flat = {}
    flat['uid'] = data.get('uid', '')
    flat['age'] = data.get('age', '')
    flat['gender'] = data.get('gender', '')
    flat['major'] = data.get('major', '')
    for i in range(1, 4):
        flat[f'food{i}'] = data.get(f'food{i}', '')
        for t in ['sweet', 'sour', 'salty', 'bitter', 'umami', 'astringent']:
            flat[f'food{i}_{t}'] = data.get(f'food{i}_{t}', '0')
        flat[f'match{i}'] = data.get(f'match{i}', '')
    flat['surprise'] = data.get('surprise', '')
    flat['useful'] = data.get('useful', '')
    flat['scene'] = data.get('scene', '')
    flat['scene_other'] = data.get('scene_other', '')
    flat['ability'] = data.get('ability', '')
    flat['ability_other'] = data.get('ability_other', '')
    flat['score'] = data.get('score', '')
    flat['feeling'] = data.get('feeling', '')
    flat['suggestion'] = data.get('suggestion', '')
    flat['timestamp'] = data.get('timestamp', datetime.now().isoformat())
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csvmod.DictWriter(f, fieldnames=list(flat.keys()))
        if not exists:
            w.writeheader()
        w.writerow(flat)
    return {'status': 'ok', 'msg': '已收集'}


@app.post('/api/ocr')
async def ocr_nutrition_label(file: UploadFile = File(...)):
    """上传营养成分表照片，OCR 识别并解析为结构化营养数据"""
    from nutrition_ocr import parse_from_image
    data = await file.read()
    if not data:
        raise HTTPException(400, '文件为空')
    try:
        lines, parsed = parse_from_image(data)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {
        'raw_lines': lines,
        'nutrition': parsed,
        'note': '识别结果按每100g计；能量已自动将千焦换算为千卡(kcal)',
    }
