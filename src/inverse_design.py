# -*- coding: utf-8 -*-
"""
逆向配方设计: 给定目标味觉向量, 在食品库中搜索最优食材组合与克数
算法: 场景约束食材池 -> 随机搜索(无重复组合 + Dirichlet 配比)
      -> 目标距离 + 低钠/低糖/调味品占比罚分排序
      -> 批量局部配比精修 -> 多样性挑选 Top-K
以训练好的 TastePredictionModel 作为味觉评估器, 全程无需额外训练,
随机种子固定, 结果可复现。
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'food_database.csv'

LOW_SODIUM_LIMIT = 120.0   # GB 28050 低钠声称: <=120 mg/100g
LOW_SUGAR_LIMIT = 5.0      # GB 28050 低糖声称: <=5 g/100g
MAX_DIST = (6 * 100) ** 0.5  # 六维味觉空间最大欧氏距离 ~24.49

SHOW_NUTRITION = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium']

# 咸口/油脂类调味品, 在饮品类场景中排除
SAVORY_CONDIMENTS = ['食盐', '酱油', '食醋', '味精', '花椒', '八角', '孜然',
                     '橄榄油', '花生油', '香油', '蚝油', '鱼露', '豆瓣酱',
                     '番茄酱', '蛋黄酱', '沙拉酱', '黑胡椒', '白胡椒',
                     '辣椒粉', '咖喱粉', '桂皮', '姜黄粉', '芥末酱', '菜籽油']

# 预包装/即食成品食品: 不作为逆向设计的烹饪/调配原料
# (逆向设计要输出"菜谱", 不是把现成零食饮料混在一起)
PROCESSED_FOODS = [
    # 碳酸/功能/调配型饮料
    '可乐', '运动饮料', '能量饮料', '气泡水', '苏打水',
    '酸奶饮品', '椰汁饮料', '拿铁咖啡', '白酒', '威士忌', '香槟',
    # 即食成品零食
    '薯片', '饼干', '蛋糕', '冰淇淋', '果冻', '方便面', '爆米花',
    '牛肉粒', '莲蓉月饼', '黑芝麻汤圆', '能量棒', '曲奇饼干', '甜甜圈',
    '牛角包', '芝士披萨', '汉堡', '炸鸡', '热狗', '寿司', '春卷',
    '炒饭', '辣条', '火腿肠', '海苔脆片', '锅巴', '蛋黄派', '酸奶果粒',
]

# 各场景的原料白名单类别 (其余类别排除)
SCENARIO_CATS = {
    '通用': ['水果', '主食', '肉禽水产', '蛋奶豆类', '蔬菜菌藻',
             '坚果种子', '饮料酒水', '调味品'],
    '饮品·甜品': ['水果', '蛋奶豆类', '坚果种子', '饮料酒水', '调味品'],
    '菜肴·主食': ['主食', '肉禽水产', '蛋奶豆类', '蔬菜菌藻',
                 '坚果种子', '调味品'],
}

# 应用场景: 限定候选食材池 + 调味品合计占比上限, 保证 culinary 合理性
SCENARIOS = {
    '通用': {'include_cats': SCENARIO_CATS['通用'],
             'exclude_names': PROCESSED_FOODS, 'cond_cap': 0.20},
    '饮品·甜品': {'include_cats': SCENARIO_CATS['饮品·甜品'],
                'exclude_names': SAVORY_CONDIMENTS + PROCESSED_FOODS,
                'cond_cap': 0.30},
    '菜肴·主食': {'include_cats': SCENARIO_CATS['菜肴·主食'],
                'exclude_names': PROCESSED_FOODS, 'cond_cap': 0.15},
}


class InverseDesigner:
    """逆向配方设计器"""

    def __init__(self, model, df):
        self.model = model
        self.df = df.reset_index(drop=True)
        self.names = self.df['name'].tolist()
        self.cats = self.df['category'].tolist()
        self.X = self.df[model.feature_cols].to_numpy(dtype=float)  # (N,16)
        self.n = len(self.df)
        self.na_i = model.feature_cols.index('sodium')
        self.su_i = model.feature_cols.index('sugar')
        self.is_cond = np.array([c == '调味品' for c in self.cats])
        # 纯油脂类(脂肪>=60g/100g: 花生油/黄油等)在任何场景都不应作主料
        self.is_fat = self.X[:, model.feature_cols.index('fat')] >= 60.0
        # 添加糖(白糖/蜂蜜): 单独限制占比, 避免"白糖当饭吃"
        self.is_added_sugar = np.isin(self.names, ['白糖', '蜂蜜'])
        # 类别编码数组, 用于类别一致性罚分
        self.cat_arr = np.array(self.cats)

    # ---------------- 批量预测 ----------------
    def _predict_batch(self, X):
        Xs = self.model.scaler.transform(X)
        return np.clip(self.model.model.predict(Xs), 0, 10)

    def _mix(self, idx, w):
        """idx (M,k), w (M,k) -> 每100g混合营养 (M,16)"""
        mixed = (w[:, :, None] * self.X[idx]).sum(axis=1)
        return mixed / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)

    def _score(self, pred, mix, idx, w, target, low_sodium, low_sugar, cond_cap):
        """目标距离 + 约束罚分(仅用于排序, 不用于展示)"""
        d = np.sqrt(((pred - target) ** 2).sum(axis=1))
        pen = np.zeros(len(d))
        if low_sodium:
            pen += (np.maximum(0.0, mix[:, self.na_i] - LOW_SODIUM_LIMIT)
                    / LOW_SODIUM_LIMIT) ** 2 * 25.0
        if low_sugar:
            pen += (np.maximum(0.0, mix[:, self.su_i] - LOW_SUGAR_LIMIT)
                    / LOW_SUGAR_LIMIT) ** 2 * 25.0
        fat_share = (w * self.is_fat[idx]).sum(axis=1)
        pen += (np.maximum(0.0, fat_share - 0.25) / 0.25) ** 2 * 8.0
        # 添加糖(白糖/蜂蜜)占比上限 20%, 避免白糖当主料
        sugar_share = (w * self.is_added_sugar[idx]).sum(axis=1)
        pen += (np.maximum(0.0, sugar_share - 0.20) / 0.20) ** 2 * 12.0
        if cond_cap is not None:
            cond_share = (w * self.is_cond[idx]).sum(axis=1)
            pen += (np.maximum(0.0, cond_share - cond_cap) / cond_cap) ** 2 * 6.0
        # 类别一致性: 避免"水果+蔬菜"或"水果+主食"等不合理混搭
        has_fruit = np.isin(self.cat_arr[idx], ['水果']).sum(axis=1) > 0
        has_veg = np.isin(self.cat_arr[idx], ['蔬菜菌藻']).sum(axis=1) > 0
        has_staple = np.isin(self.cat_arr[idx], ['主食']).sum(axis=1) > 0
        pen += (has_fruit & has_veg).astype(float) * 6.0
        pen += (has_fruit & has_staple).astype(float) * 6.0
        return d + pen, d

    # ---------------- 搜索 ----------------
    def _sample(self, rng, k, m, pool):
        """从候选池中抽 m 组互不重复的 k 食材组合 + Dirichlet 随机配比"""
        order = rng.random((m, len(pool))).argpartition(k - 1, axis=1)[:, :k]
        w = rng.dirichlet(np.ones(k) * 1.5, size=m)
        return pool[order], w

    def _refine_batch(self, rng, idx, w, target, low_sodium, low_sugar,
                      cond_cap, iters=30):
        """整组候选并行配比扰动精修(每次迭代一次批量预测)"""
        m, k = w.shape
        best_w = w.copy()
        mix = self._mix(idx, best_w)
        pred = self._predict_batch(mix)
        best_s, best_d = self._score(pred, mix, idx, best_w, target,
                                     low_sodium, low_sugar, cond_cap)
        rows = np.arange(m)
        for _ in range(iters):
            j = rng.integers(0, k, size=m)
            new_w = best_w.copy()
            new_w[rows, j] = np.maximum(
                0.05, new_w[rows, j] + rng.normal(0, 0.06, size=m))
            new_w /= new_w.sum(axis=1, keepdims=True)
            mix = self._mix(idx, new_w)
            pred = self._predict_batch(mix)
            s, dd = self._score(pred, mix, idx, new_w, target,
                                low_sodium, low_sugar, cond_cap)
            better = s < best_s - 1e-6
            best_w[better] = new_w[better]
            best_s = np.where(better, s, best_s)
            best_d = np.where(better, dd, best_d)
        return best_w, best_s, best_d

    @staticmethod
    def _grams(w, total):
        """配比 -> 保留1位小数的实际克数(单食材>=10g)"""
        g = np.maximum(10.0, np.round(w * total, 1))
        diff = round(float(total - g.sum()), 1)
        g[int(np.argmax(g))] = round(float(g[int(np.argmax(g))] + diff), 1)
        return g

    # ---------------- 主入口 ----------------
    def design(self, target, total_grams=300, k='auto', scenario='通用',
               low_sodium=False, low_sugar=False, n_results=3,
               seed=42, per_k=8000, refine_top=40, refine_iters=30):
        """
        target: dict, 键为 sweet/sour/salty/bitter/umami/astringent, 值 0-10
        scenario: SCENARIOS 键名, 限定候选食材池保证组合合理
        返回: list[dict], 按匹配度排序的 n_results 个多样化方案
        """
        rng = np.random.default_rng(seed)
        tvec = np.array([float(target.get(c, 0))
                         for c in self.model.target_cols]).reshape(1, -1)
        ks = [2, 3, 4] if str(k) == 'auto' else [int(k)]
        sc = SCENARIOS.get(scenario, SCENARIOS['通用'])
        # 通用场景: 根据目标味觉自动推断甜口/咸口, 套用对应类别白名单,
        # 避免"西瓜炒肉"这类跨风味方向的不合理混搭
        if scenario == '通用':
            sweet_v = float(target.get('sweet', 0))
            savory_v = max(float(target.get('salty', 0)),
                           float(target.get('umami', 0)))
            if sweet_v >= savory_v:
                cats = SCENARIO_CATS['饮品·甜品']
                exc = SAVORY_CONDIMENTS + PROCESSED_FOODS
                cond_cap = 0.30
            else:
                cats = SCENARIO_CATS['菜肴·主食']
                exc = PROCESSED_FOODS
                cond_cap = 0.15
        else:
            cats = sc['include_cats']
            exc = sc['exclude_names']
            cond_cap = sc['cond_cap']
        pool = np.array([i for i in range(self.n)
                         if self.cats[i] in cats
                         and self.names[i] not in exc])

        refined = []
        for kk in ks:  # 不同 k 组合列数不同, 分组采样评分后合并
            idx_k, w_k = self._sample(rng, kk, per_k, pool)
            mix = self._mix(idx_k, w_k)
            pred = self._predict_batch(mix)
            score, _ = self._score(pred, mix, idx_k, w_k, tvec,
                                   low_sodium, low_sugar, cond_cap)
            sel = np.argsort(score)[:refine_top]
            bw, bs, bd = self._refine_batch(
                rng, idx_k[sel], w_k[sel], tvec, low_sodium, low_sugar,
                cond_cap, iters=refine_iters)
            for r in range(len(sel)):
                refined.append((float(bs[r]), float(bd[r]),
                                idx_k[sel][r], bw[r]))
        refined.sort(key=lambda x: x[0])

        # 多样性挑选: 两两方案共享食材不超过1种, 不够则放宽
        picked = []
        for share_max in (1, 2, 99):
            for bs, bd, idx, w in refined:
                if len(picked) >= n_results:
                    break
                if all(len(set(idx.tolist()) & set(p[2].tolist())) <= share_max
                       for p in picked):
                    picked.append((bs, bd, idx, w))
            if len(picked) >= n_results:
                break

        results = []
        for bs, bd, idx, w in picked[:n_results]:
            grams = self._grams(w, total_grams)
            items = [{'name': self.names[i], 'grams': round(float(g), 1)}
                     for i, g in zip(idx.tolist(), grams.tolist())]
            items.sort(key=lambda x: -x['grams'])
            w_final = grams.astype(float) / grams.sum()
            mix1 = self._mix(idx[None], w_final[None])[0]
            pred1 = self._predict_batch(mix1[None])[0]
            results.append({
                'items': items,
                'total_grams': round(float(grams.sum()), 1),
                'predicted_taste': {c: round(float(pred1[j]), 2)
                                    for j, c in enumerate(self.model.target_cols)},
                'match_score': round(100 * max(0.0, 1 - bd / MAX_DIST), 1),
                'nutrition_per100g': {c: round(float(
                    mix1[self.model.feature_cols.index(c)]), 2)
                    for c in SHOW_NUTRITION},
            })
        return results


def design_recipes(model, df, target, **kwargs):
    """便捷入口: 复用已训练的模型与数据库"""
    return InverseDesigner(model, df).design(target, **kwargs)


if __name__ == '__main__':
    import sys, time
    sys.path.insert(0, str(Path(__file__).parent))
    from taste_model import TastePredictionModel

    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    model = TastePredictionModel()
    model.train(str(DATA_PATH))
    d = InverseDesigner(model, df)

    cases = [('清甜水果饮', {'sweet': 8, 'sour': 2}, '饮品·甜品', {}),
             ('咸鲜低钠汤', {'salty': 6, 'umami': 7}, '菜肴·主食',
              {'low_sodium': True}),
             ('微甜低糖零食', {'sweet': 5, 'salty': 1}, '通用',
              {'low_sugar': True})]
    for name, tgt, sc, kw in cases:
        t0 = time.time()
        rs = d.design(tgt, scenario=sc, seed=42, **kw)
        print(f'\n== {name} [{sc}] (耗时 {time.time()-t0:.1f}s)')
        for r in rs:
            its = ' + '.join(f"{i['name']}{i['grams']}g" for i in r['items'])
            nu = r['nutrition_per100g']
            print(f"  匹配{r['match_score']:>5}% | {its}")
            print(f"         钠{nu['sodium']:.0f}mg 糖{nu['sugar']:.1f}g "
                  f"{nu['calories']:.0f}kcal | {r['predicted_taste']}")
