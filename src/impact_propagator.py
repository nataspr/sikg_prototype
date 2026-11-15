from collections import deque  # двусторонняя очередь для обхода в ширину
import networkx as nx 
# СДЕЛАНО ПО СТАТЬЕ SKIG! 
# Функция распространяет влияние от изменённых модулей по графу
def propagate_impact(
    G: nx.DiGraph,           # граф, построенный в graph_builder
    changed_modules: list,   # список имён изменённых модулей
    base_risk: float,        # базовый риск, полученный в change_analyzer
    max_depth: int = 3,      # максимальная глубина обхода
    decay: float = 0.6,      # коэффициент затухания на каждом шаге - для уменьшения оценки влияния
    threshold: float = 0.1,  # порог, ниже которого влияние не распространяется
) -> dict:
    impact = {}         # словарь накопленных оценок влияния по узлам
    queue = deque()     # очередь для обхода в ширину, хранит пары узел и глубина

    # Каждый изменённый модуль получает базовый риск и кладётся в очередь на нулевой глубине
    for mod in changed_modules:
        if G.has_node(mod):  # проверка, что такой модуль вообще есть в графе
            impact[mod] = base_risk
            queue.append((mod, 0))  # узел и его глубина от источника

    # Основной цикл обхода в ширину
    while queue:
        node, depth = queue.popleft()  # узел из начала очереди
        if depth >= max_depth:
            continue  # если достигли предельной глубины, дальше не идём

        current_impact = impact.get(node, 0.0)  # текущая оценка влияния узла

        # Перебираются все исходящие соседи узла
        for neighbor in G.successors(node):
            edge_data = G.get_edge_data(node, neighbor)  # атрибуты ребра
            edge_weight = edge_data.get("weight", 1.0)   # вес ребра, по умолчанию единица

            # Вычисляется доля влияния, доходящая до соседа: текущее влияние умножается на вес ребра и на коэффициент затухания в степени глубины
            propagated = current_impact * edge_weight * (decay ** depth)

            # Если доля выше порога, сосед обновляется и попадает в очередь
            if propagated > threshold:
                old = impact.get(neighbor, 0.0)  # прежнее значение
                # максимум, чтобы не понижать уже установленную оценку
                impact[neighbor] = max(old, propagated)
                queue.append((neighbor, depth + 1))

    # Из накопленных оценок оставляются только узлы типа test, т к для них будет приоретизация
    return {
        n: score
        for n, score in impact.items()
        if G.nodes[n].get("type") == "test"
    }