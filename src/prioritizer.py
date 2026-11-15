import os
import pandas as pd

from src.graph_builder import build_graph
from src.change_analyzer import classify_commit, get_changed_files, CHANGE_TYPES
from src.impact_propagator import propagate_impact
from src.lgbm_agent_modified import LightGBMAgent

# на вход путь к репозиторию, хеш коммита и тип агента, на выход отсортированный список тестов и словарь оценок
def run_pipeline(repo_path: str, commit_hash: str, agent_type: str = "original"):
    # граф знаний о проекте
    G = build_graph(repo_path)

    # анализ коммита
    change_type = classify_commit(repo_path, commit_hash)  # тип изменения
    base_risk = CHANGE_TYPES.get(change_type, 0.5)         # базовый риск по типу
    changed_files = get_changed_files(repo_path, commit_hash)  # изменённые файлы

    # Имена модулей из имён файлов без расширения py
    changed_modules = [
        os.path.splitext(os.path.basename(f))[0] for f in changed_files
    ]

    # распрстранение влияния по графу
    impact_scores = propagate_impact(G, changed_modules, base_risk)

    # Если после распространения нет ни одного затронутого теста,
    # возвращается пустой результат.
    if not impact_scores:
        return [], {}

    # Приоритизация
    if agent_type == "original":
        # сортировка по убыванию оценки, rl агент
        ranked = sorted(
            impact_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [t for t, _ in ranked], impact_scores
    else:

        agent = LightGBMAgent()
        # Формируются метаданные тестов, в проекте они берутся из истории
        test_metadata = {
            t: {
                "fail_rate": 0.3,           # доля падений в прошлом
                "age": 10,                  # возраст теста
                "change_type": change_type, # тип текущего изменения
            }
            for t in impact_scores
        }

        # таблица признаков
        df = agent.prepare_features(impact_scores, test_metadata)

        # Тест считается упавшим, если его оценка влияния выше медианы.
        median_impact = df["impact_score"].median()
        labels = (df["impact_score"] > median_impact).astype(int).values

        try:
            # Модель обучается на текущих данных и сразу предсказывает вероятности
            agent.train(df, labels)
            probs = agent.predict_proba(df)
        except Exception:            
            probs = df["impact_score"].values # Если данных слишком мало для обучения, используется оценка влияния

        # Итоговый рейтинг сортируется по убыванию предсказанной вероятности
        results = list(zip(df["test_name"].tolist(), probs.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        return [t for t, _ in results], dict(results)