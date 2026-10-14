import sys
sys.path.insert(0, r'c:\Users\云士博\AppData\Roaming\Trae CN\ModularData\ai-agent\work-mode-projects\6acca47a9de965489a99c0d3\nutritongue\src')
from taste_model import TastePredictionModel, NutritionAnalyzer, FoodEvaluator

DATA = r'c:\Users\云士博\AppData\Roaming\Trae CN\ModularData\ai-agent\work-mode-projects\6acca47a9de965489a99c0d3\nutritongue\data\food_database.csv'

model = TastePredictionModel()
metrics = model.train(DATA)
print('=== 模型评估 ===')
for k, v in metrics.items():
    print(f'{k}: {v}')

print()
print('=== 预测验证 ===')
tests = {
    '香蕉': {'calories': 89, 'protein': 1.1, 'carbs': 22.8, 'fat': 0.3, 'fiber': 2.6, 'sugar': 12.2, 'sodium': 1, 'potassium': 358, 'calcium': 5, 'iron': 0.26, 'vitamin_c': 8.7, 'organic_acid': 0.3, 'polyphenol': 30, 'caffeine': 0, 'alkaloid': 0, 'tannin': 3},
    '黑咖啡': {'calories': 2, 'protein': 0.3, 'carbs': 0, 'fat': 0, 'fiber': 0, 'sugar': 0, 'sodium': 2, 'potassium': 49, 'calcium': 2, 'iron': 0, 'vitamin_c': 0, 'organic_acid': 0.1, 'polyphenol': 120, 'caffeine': 40, 'alkaloid': 0, 'tannin': 150},
    '酱油': {'calories': 63, 'protein': 5.6, 'carbs': 6.4, 'fat': 0.1, 'fiber': 0.1, 'sugar': 1.8, 'sodium': 5777, 'potassium': 253, 'calcium': 18, 'iron': 2.0, 'vitamin_c': 0, 'organic_acid': 0.2, 'polyphenol': 10, 'caffeine': 0, 'alkaloid': 0, 'tannin': 5},
}
analyzer = NutritionAnalyzer()
evaluator = FoodEvaluator()
for name, n in tests.items():
    taste = model.predict(n)
    hs = analyzer.health_score(n)
    print(f'{name}: {taste} | 健康{hs} | {evaluator.taste_description(taste)}')
