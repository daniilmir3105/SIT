import os
import numpy as np


def resize_seismograms(folder_path, target_shape=(1150, 800)):
    """
Приводит все сейсмограммы в указанной папке к заданному размеру.

:param folder_path: Путь к папке с .npy файлами.
:param target_shape: Кортеж с целевым размером (высота, ширина).
    """
    height, width = target_shape
    updated_files = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)

            # Загружаем данные из файла
            data = np.load(file_path)

            # Проверяем размеры сейсмограммы
            if data.shape[0] != height or data.shape[1] != width:
                print(f"Resizing {file_name}: original shape {data.shape}")

                # Приведение к целевому размеру
                resized_data = data[:height, :width]

                # Сохраняем обрезанную сейсмограмму обратно
                np.save(file_path, resized_data)
                updated_files += 1

    print(f"Обновлено файлов: {updated_files}/{len(os.listdir(folder_path))}")


if __name__ == "__main__":
    # Укажите путь к папке с сейсмограммами
    # folder_with_seismograms = "path_to_your_subsampled_folder"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_with_seismograms = os.path.join(current_dir, "data")
    # folder_with_seismograms = os.path.join(current_dir, "subsampled")
    # Приводим все файлы в папке к размеру (1150, 800)
    resize_seismograms(folder_with_seismograms)
    