import os
import subprocess
import sys
from src.prioritizer import run_pipeline
from src.metrics import precision_recall_f1, apfd

# возврат списка последних коммитов репозитория
def get_commits(repo_path: str, n: int = 5):
    # git log выводит только хеши коммитов, ограничение по числу задаётся через n
    out = subprocess.check_output(
        ["git", "-C", repo_path, "log", "--pretty=format:%H", "-n", str(n)],
        text=True,
    )
    return [h for h in out.splitlines() if h.strip()]

def main():
    # Путь к репозиторию isort
    repo_path = "external/isort"

    if not os.path.isdir(repo_path):
        print(f"Ошибка: {repo_path} не найден. Сделайте git submodule update --init")
        sys.exit(1)

    # пять последних коммитов для эксперимента
    commits = get_commits(repo_path, n=5)

    # Перебираются все коммиты по очереди
    for commit in commits:
        print(f"\n{'='*70}\nКоммит: {commit[:10]}")

        # оригинальная версия
        ranked_orig, _ = run_pipeline(repo_path, commit, "original")
        # изменённая версия
        ranked_mod, _ = run_pipeline(repo_path, commit, "modified")

        # Если оригинальная версия не нашла изменений в py файлах, коммит пропускается
        if not ranked_orig:
            print("  Нет изменений в .py файлах, пропускаем")
            continue

        # эталон упавших тестов для демонстрации-первые три теста из оригинального рейтинга
        # В реальном эксперименте это результат запуска pytest
        failed = set(ranked_orig[:3])

        # метрики для оригинальной версии
        p_o, r_o, f1_o = precision_recall_f1(ranked_orig, failed, k=5)
        apfd_o = apfd(ranked_orig, failed)
        # метрики для изменённой версии
        p_m, r_m, f1_m = precision_recall_f1(ranked_mod, failed, k=5)
        apfd_m = apfd(ranked_mod, failed)

        print(f"  Original (RL):     P={p_o:.2f} R={r_o:.2f} F1={f1_o:.2f} APFD={apfd_o:.3f}")
        print(f"  Modified (LightGBM): P={p_m:.2f} R={r_m:.2f} F1={f1_m:.2f} APFD={apfd_m:.3f}")

if __name__ == "__main__":
    main()