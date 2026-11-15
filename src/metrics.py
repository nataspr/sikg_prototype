# на вход отсортированный список тестов, множество реально упавших тестов
def precision_recall_f1(ranked_tests: list, failed_set: set, k: int):
    top_k = ranked_tests[:k]

    # Считаются истинно положительные случаи=тесты в топе и в множестве реально упавших
    tp = sum(1 for t in top_k if t in failed_set)

    # Точность равна доле упавших среди первых k
    precision = tp / k if k > 0 else 0.0

    # Полнота равна доле найденных упавших среди всех реально упавших
    recall = tp / len(failed_set) if failed_set else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1 # точность, полнота и F мера для первых k тестов в рейтинге


# Функция вычисляет метрику APFD (средняя доля обнаруженных ошибок, лучше 1)
def apfd(ranked_tests: list, failed_set: set):
    n = len(ranked_tests)  # общее число тестов
    m = len(failed_set)    # число реально упавших тестов
    if n == 0 or m == 0:
        return 0.0

    # Находятся позиции упавших тестов в рейтинге
    positions = [i + 1 for i, t in enumerate(ranked_tests) if t in failed_set]

    # Если ни один упавший тест не попал в рейтинг, метрика равна нулю
    if not positions:
        return 0.0

    return 1.0 - (sum(positions) / (n * m)) + (1.0 / (2 * n))