import lightgbm as lgb
import pandas as pd
import numpy as np

# Модель градиентного бустинга предсказывает вероятность того, что тест упадёт
class LightGBMAgent:
    def __init__(self):
        self.model = None          # сюда будет сохранена обученная модель
        self.feature_names = None  # список имён признаков

    # формирование таблицы признаков из словаря оценок влияния и метаданных тестов
    def prepare_features(self, impact_scores: dict, test_metadata: dict):
        rows = []
        # Перебираются все тесты из словаря оценок влияния
        for test_name, impact in impact_scores.items():
            # метаданные конкретного теста, если они есть
            meta = test_metadata.get(test_name, {})

            # одна строка признаков
            rows.append({
                "test_name": test_name,  # имя теста, идентификатор!
                "impact_score": impact,  # оценка влияния из графа
                "fail_rate": meta.get("fail_rate", 0.3),  # историческая частота падений
                "age": meta.get("age", 10),               # возраст теста
                # Признаки типа изменения
                "change_type_bugfix": 1 if meta.get("change_type") == "BUG_FIX" else 0,
                "change_type_feature": 1 if meta.get("change_type") == "FEATURE_ADDITION" else 0,
                "change_type_refactor": 1 if meta.get("change_type") == "REFACTORING_LOGIC" else 0,
            })

        return pd.DataFrame(rows)

    # обучение модели LightGBM на подготовленной таблице и метках
    def train(self, df: pd.DataFrame, labels: np.ndarray):
        self.feature_names = [c for c in df.columns if c != "test_name"] # Имена признаков
        X = df[self.feature_names]  # матрица признаков

        # классификатор градиентного бустинга
        self.model = lgb.LGBMClassifier(
            n_estimators=100,     # сто деревьев
            max_depth=5,          # глубина
            learning_rate=0.1,    # скорость обучения
            random_state=42,      # для воспроизводимости
            verbose=-1,
        )

        self.model.fit(X, labels)

    # вероятность падения для каждого теста
    def predict_proba(self, df: pd.DataFrame):
        X = df[self.feature_names]  # матрица признаков
        # вероятность fail_rate (метка=1)
        return self.model.predict_proba(X)[:, 1]