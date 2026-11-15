import numpy as np
from collections import defaultdict    # словарь, создающий значение по умолчанию

# механизм адаптирован из статьи SIKG
# Агент запоминает ценность каждого действия в каждом состоянии и постепенно уточняет эти ценности по мере получения наград
class OriginalRLAgent:
    # Конструктор принимает число тестов и гиперпараметры из статьи пример
    def __init__(self, n_tests, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha          # скорость обучения, насколько сильно обновляется Q
        self.gamma = gamma          # коэффициент дисконтирования будущих наград
        self.epsilon = epsilon      # доля случайных исследований
        self.n_tests = n_tests      # общее число доступных действий, то есть тестов

        # Q таблица хранит для каждого состояния массив оценок по числу тестов
        self.q_table = defaultdict(lambda: np.zeros(n_tests)) # defaultdict создаёт нулевой массив при первом обращении к новому состоянию
        self.state = None  # текущее состояние агента

    # Метод фиксации текущего состояния(набор изменённых модулей)
    def set_state(self, changed_modules):
        self.state = tuple(sorted(changed_modules))

    # Метод выбора действий по эпсилон жадной стратегии
    def choose_action(self) -> int:
        # С вероятностью эпсилон выбирается случайный тест
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_tests)
        # Иначе берутся оценки Q для текущего состояния
        q_values = self.q_table[self.state]        
        if np.all(q_values == 0):
            return np.random.randint(self.n_tests) # Если агент ещё не обучен и все оценки нулевые, выбирается случайный тест

        # Иначе возвращается индекс теста с максимальной оценкой Q
        return int(np.argmax(q_values))

    # Метод обновляет Q значение после получения награды
    def update(self, test_idx, reward, next_state_modules):
        old_q = self.q_table[self.state][test_idx] # Старое значение Q для текущего состояния и выбранного действия
        # ключ следующего состояния
        next_state = tuple(sorted(next_state_modules))
        # Берётся максимальная оценка в следующем состоянии
        next_max_q = (
            np.max(self.q_table[next_state])
            if self.q_table[next_state].size
            else 0.0
        )

        # новое значение = старое + шаг обучения * (целевое значение - старое)
        new_q = old_q + self.alpha * (
            reward + self.gamma * next_max_q - old_q
        )
        # Обновлённое значение записывается обратно в таблицу
        self.q_table[self.state][test_idx] = new_q