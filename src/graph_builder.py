import ast                    # для разбора Python-кода в AST 
import os                     # для работы с путями
import subprocess             # для вызова git из Python
from collections import defaultdict  # dict со значениями по умолчанию

import networkx as nx         # библиотека графов

# сбор всех питоновских файлов проекта кроме тестовых
# repo_path — путь к корню репозитория пример 'external/isort'
# вернет список абсолютных путей к .py файлам
def collect_python_files(repo_path: str):
    python_files = []
    # os.walk рекурсивно обходит все подпапки
    for root, dirs, files in os.walk(repo_path):
        # Пропуск .git и папки с тестами
        if ".git" in root or "tests" in root or "test" in root.lower():
            continue
        for f in files:
            if f.endswith(".py"):
                # os.path.join склеивает пути кроссплатформенно
                python_files.append(os.path.join(root, f))
    return python_files

# Собирает файлы с тестами: test_*.py или *_test.py
def collect_test_files(repo_path: str):
    test_files = []
    for root, dirs, files in os.walk(repo_path):
        if ".git" in root:
            continue
        for f in files:
            if f.startswith("test_") or f.endswith("_test.py"):
                test_files.append(os.path.join(root, f))
    return test_files

# Функция читает файл и через модуль ast достаёт имена всех импортируемых модулей
# Пример, из строки from foo.bar import baz в результат попадёт foo
def extract_imports(file_path: str):
    # Файл открывается с явной кодировкой utf-8, иначе на Windows возникают ошибки
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            # Если файл не удалось разобрать, возвращается пустое множество
            return set()

    imports = set()
    # ast.walk обходит все узлы синтаксического дерева
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # Для строки import foo, bar берётся первая часть имени до точки
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Для строки from foo.bar import baz берётся foo
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


# Главная функция модуля. Она возвращает направленный граф из библиотеки networkx
# У каждого узла графа есть атрибут type, он равен либо module, либо test, а также путь к файлу
def build_graph(repo_path: str) -> nx.DiGraph:
    # DiGraph это направленный граф, его рёбра имеют направление
    G = nx.DiGraph()

    # 1 Создаются узлы для модулей приложения
    for py_file in collect_python_files(repo_path):
        # Имя модуля берётся из имени файла без расширения py
        module_name = os.path.splitext(os.path.basename(py_file))[0]
        # Если узел уже есть, add_node просто обновит его атрибуты
        G.add_node(module_name, type="module", path=py_file)

    # 2 Создаются узлы для тестовых функций и рёбра TESTS
    for test_file in collect_test_files(repo_path):
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError):
            continue

        # Ищутся все функции, имя которых начинается с test_
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                # add_node создаёт узел для найденной тестовой функции
                G.add_node(node.name, type="test", path=test_file)

                # Далее проверяется, что импортирует тестовый файл
                # и если такой модуль уже есть в графе, создаётся ребро
                for imp in extract_imports(test_file):
                    if G.has_node(imp):
                        # add_edge принимает источник, цель и атрибуты ребра
                        G.add_edge(node.name, imp, type="TESTS", weight=1.0)

    # 3 Добавляются рёбра CO_CHANGE, которые отражают историческую связь
    # Для этого читается git log и находятся пары файлов, менявшиеся в одном коммите
    try:
        log = subprocess.check_output(
            ["git", "-C", repo_path, "log", "--name-only", "--pretty=format:%H"],
            text=True,
        )
    except subprocess.CalledProcessError:
        # Если папка не является git репозиторием, возвращается граф без исторических связей
        return G

    co_change = defaultdict(int)

    current_commit = None
    files_in_commit = []

    # Вывод git log разбирается построчно
    for line in log.splitlines():
        if line.strip() == "":
            # Пустая строка означает конец текущего коммита
            if current_commit and len(files_in_commit) > 1:
                # Каждая пара файлов внутри коммита увеличивает счётчик совместных изменений
                for i in range(len(files_in_commit)):
                    for j in range(i + 1, len(files_in_commit)):
                        pair = tuple(sorted([files_in_commit[i], files_in_commit[j]]))
                        co_change[pair] += 1
            current_commit = None
            files_in_commit = []
        elif len(line) == 40:
            # Хеш коммита в SHA-1 всегда состоит из 40 символов
            current_commit = line
        else:
            # Все остальные строки это имена изменённых файлов
            files_in_commit.append(line.strip())

    # Полученные пары превращаются в рёбра между модулями
    for (f1, f2), count in co_change.items():
        m1 = os.path.splitext(os.path.basename(f1))[0]
        m2 = os.path.splitext(os.path.basename(f2))[0]
        # Ребро создаётся только когда оба узла уже есть в графе
        if G.has_node(m1) and G.has_node(m2):
            if G.has_edge(m1, m2):
                # Если ребро уже есть, вес увеличивается
                G[m1][m2]["weight"] += count * 0.1
            else:
                G.add_edge(m1, m2, type="CO_CHANGE", weight=count * 0.1)

    return G