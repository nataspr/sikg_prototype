import networkx as nx
import pytest
from src.impact_propagator import propagate_impact
from src.metrics import precision_recall_f1, apfd
from src.rl_agent_original import OriginalRLAgent

# распространение влияния работает на графе из двух узлов??
def test_propagate_impact_simple():
    # направленный граф
    G = nx.DiGraph()
    # Добавляется узел модуля и узел теста
    G.add_node("m1", type="module")
    G.add_node("test_1", type="test")

    # Между тестом и модулем создаётся ребро типа TESTS с весом единица
    # ОТ теста к модулю node.name = тест, imp = модуль
    G.add_edge("test_1", "m1", type="TESTS", weight=1.0)

    # распространение влияния от модуля m1
    result = propagate_impact(G, ["m1"], base_risk=1.0, max_depth=2, decay=0.6)

    # тест попал в результат?
    assert "test_1" in result

    # оценка влияния положительна?
    assert result["test_1"] > 0

# Тест проверяет корректность метрик точности и полноты
def test_precision_recall_f1():
    ranked = ["t1", "t2", "t3", "t4"]  # рейтинг тестов
    failed = {"t1", "t3"}              # реально упавшие тесты

    # метрики для первых двух позиций
    p, r, f1 = precision_recall_f1(ranked, failed, k=2)

    # В первых двух тестах один упавший, точность половина
    assert p == 0.5
    # Из двух упавших найден один, полнота половина
    assert r == 0.5
    # F мера должна оказаться между нулём и единицей
    assert 0 < f1 < 1

# идеальный рейтинг даёт высокий APFD
def test_apfd_perfect_ranking():
    # n = 10, m = 2, оба упавших на позициях 1 и 2
    ranked = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]
    failed = {"t1", "t2"}
    score = apfd(ranked, failed)
    # APFD = 1 - (1+2)/(10*2) + 1/(2*10) = 1 - 0.15 + 0.05 = 0.9
    assert score > 0.8

# Тест проверяет, что RL агент корректно обновляет Q значения, оценка выше 0
def test_rl_agent_update():
    agent = OriginalRLAgent(n_tests=3)  # агент на три возможных действия

    # установка состояния
    agent.set_state(["m1"])
    # положительная награда к действию с индексом ноль
    agent.update(test_idx=0, reward=1.0, next_state_modules=["m1", "m2"])

    assert agent.q_table[("m1",)][0] > 0