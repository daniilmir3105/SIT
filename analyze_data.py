import os
import segyio
import numpy as np


def analyze_segy_files(data_dir):
    # Проходим по всем файлам в указанной папке
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.segy') or file_name.endswith('.sgy'):
            file_path = os.path.join(data_dir, file_name)
            try:
                with segyio.open(file_path, "r", ignore_geometry=True) as f:
                    # Проверяем, доступны ли трассы
                    if not f.trace:
                        raise ValueError("Файл не содержит данных трасс.")

                    # Проверяем размеры с помощью samples и ilines/xlines
                    try:
                        data = segyio.tools.cube(f)  # Пробуем загрузить как 3D массив
                        if len(data.shape) == 3:  # Если форма корректна
                            print(f"Файл: {file_name}")
                            print(f"Тип: 3D сейсмограмма")
                            print(f"Размер: {data.shape[0]}×{data.shape[1]}×{data.shape[2]}\n")
                        else:
                            print(f"Файл: {file_name}")
                            print(f"Тип: 2D сейсмограмма\n")
                    except Exception:
                        # Если `cube` не работает, пробуем вручную оценить размерности
                        num_traces = len(f.trace)
                        num_samples = len(f.samples)
                        print(f"Файл: {file_name}")
                        print(f"Тип: Возможно, 2D или некорректный формат")
                        print(f"Число трасс: {num_traces}, Число сэмплов: {num_samples}\n")
            except Exception as e:
                print(f"Не удалось обработать файл {file_name}. Ошибка: {e}\n")


if __name__ == "__main__":
    # Укажите путь к папке с файлами
    data_dir = r"C:\Users\Daniil\PycharmProjects\SIT\data"
    analyze_segy_files(data_dir)
