import os
import numpy as np

def print_seismic_size():
    """
    Выводит размеры любого файла сейсмограммы из папки 'data'.
    Предполагается, что файлы хранятся в формате .npy.
    """
    # Определяем текущую директорию, где находится исполняемый файл
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    # Проверяем, существует ли папка 'data'
    if not os.path.exists(data_dir):
        print(f"Папка '{data_dir}' не существует.")
        return

    # Находим файлы с расширением .npy в папке 'data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    if not files:
        print("В папке 'data' нет файлов с расширением .npy.")
        return

    # Выбираем первый файл и выводим его размер
    first_file = files[0]
    file_path = os.path.join(data_dir, first_file)
    try:
        seismic_data = np.load(file_path)
        print(f"Размер первой сейсмограммы ({first_file}): {seismic_data.shape}")
    except Exception as e:
        print(f"Ошибка при обработке файла {first_file}: {e}")

if __name__ == "__main__":
    print_seismic_size()
