import os
import segyio
import numpy as np


def create_patches(data_dir, output_dir, patch_size=224):
    """
    Создает патчи из SEGY/SGY файлов и сохраняет их в формате .npy.

    :param data_dir: Директория с SEGY/SGY файлами.
    :param output_dir: Директория для сохранения патчей.
    :param patch_size: Размер патча (например, 224x224, кратный 7x7).
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.segy') or file_name.endswith('.sgy'):
            file_path = os.path.join(data_dir, file_name)
            try:
                with segyio.open(file_path, "r", ignore_geometry=True) as f:
                    # Пытаемся загрузить данные как индивидуальные трассы
                    traces = [np.copy(trace) for trace in f.trace if len(trace) == len(f.samples)]
                    if not traces:
                        raise ValueError("Файл не содержит валидных трасс.")

                    # Преобразуем трассы в массив (traces x samples)
                    data = np.stack(traces, axis=0)

                    # Проверяем размерность массива
                    if len(data.shape) == 2:
                        print(f"Обработка файла: {file_name}, размер: {data.shape}")

                        # Создаем патчи из 2D данных
                        for x in range(0, data.shape[0] - patch_size + 1, patch_size):
                            for y in range(0, data.shape[1] - patch_size + 1, patch_size):
                                patch = data[x:x + patch_size, y:y + patch_size]
                                patch_name = f"{file_name}_patch_{x}_{y}.npy"
                                np.save(os.path.join(output_dir, patch_name), patch)
                    else:
                        print(f"Не удалось определить размерность файла: {file_name}")

            except Exception as e:
                print(f"Ошибка при обработке файла {file_name}: {e}")


if __name__ == "__main__":
    # Директория с исходными файлами
    data_dir = r"C:\Users\Daniil\PycharmProjects\SwinTransformer\data"

    # Директория для сохранения патчей
    output_dir = r"C:\Users\Daniil\PycharmProjects\SwinTransformer\patches"

    # Размер патча (например, 224x224, кратный 7x7)
    patch_size = 224

    # Запуск создания патчей
    create_patches(data_dir, output_dir, patch_size=patch_size)
