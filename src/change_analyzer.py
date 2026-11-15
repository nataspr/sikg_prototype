import subprocess  # модуль для запуска внешних программ, используется git
# СДЕЛАНО ПО СТАТЬЕ SKIG! значения взяты оттуда
# Словарь базовых рисков для каждого типа изменения - чем выше число, тем сильнее изменение влияет на тесты
CHANGE_TYPES = {
    "BUG_FIX": 1.0,                # исправление багов
    "REFACTORING_SIGNATURE": 0.9,  # изменение сигнатур функций
    "FEATURE_ADDITION": 0.8,       # добавление новой функциональности
    "DEPENDENCY_UPDATE": 0.7,      # обновление зависимостей
    "REFACTORING_LOGIC": 0.5,      # переписывание логики без изменения поведения
    "DOCUMENTATION": 0.1,          # правки документации, почти не влияют на тесты
}

# Функция определяет тип изменения по тексту сообщения коммита
# На вход: путь к репозиторию и хеш коммита, возвращает строку, которая является ключом словаря CHANGE_TYPES
def classify_commit(repo_path: str, commit_hash: str) -> str:
    # Запускается git log, чтобы получить только текст сообщения коммита
    msg = subprocess.check_output(
        ["git", "-C", repo_path, "log", "-1", "--pretty=%B", commit_hash],
        text=True,
    ).lower()  # нижний регистр для простоты поиска

    # Проверяются ключевые слова по приоритету
    if any(w in msg for w in ["fix", "bug", "hotfix", "issue", "patch"]):
        return "BUG_FIX"
    if any(w in msg for w in ["add", "feature", "implement", "introduce", "new"]):
        return "FEATURE_ADDITION"
    if any(w in msg for w in ["rename", "move", "signature", "argument"]):
        return "REFACTORING_SIGNATURE"
    if any(w in msg for w in ["refactor", "cleanup", "reorganize"]):
        return "REFACTORING_LOGIC"
    if any(w in msg for w in ["doc", "readme", "comment", "typo"]):
        return "DOCUMENTATION"
    if any(w in msg for w in ["bump", "upgrade", "dependency", "requirements"]):
        return "DEPENDENCY_UPDATE"
    # Если ни одно слово не подошло, возвращается средний тип
    return "REFACTORING_LOGIC"


# Функция возвращает список изменённых файлов с расширением py по пути к репозиторию и хешу коммита
def get_changed_files(repo_path: str, commit_hash: str):
    # git show с опцией name-only выводит имена изменённых файлов.
    output = subprocess.check_output(
        [
            "git", "-C", repo_path, "show",
            "--name-only", "--pretty=format:", commit_hash,
        ],
        text=True,
    )
    # Результат разбивается по строкам, пустые строки и не py файлы отбрасываются
    return [f for f in output.splitlines() if f.strip() and f.endswith(".py")]