import os
import pstats
from datetime import timedelta

BASE_DIR = r"D:\Programming\pythonProject\Work-avc-Russia\out\r_bnb_first"


def print_profiling_time(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "profile_results.prof" in files:
            prof_path = os.path.join(root, "profile_results.prof")

            # Загружаем данные профилирования
            stats = pstats.Stats(prof_path)

            # Извлекаем общее время выполнения (в секундах)
            total_time = stats.total_tt

            print(f"\nФайл: {prof_path}")
            print(f"Общее время выполнения: {total_time:.2f} сек.")


if __name__ == "__main__":
    print_profiling_time(BASE_DIR)
