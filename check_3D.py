import os
import segyio

def check_if_3d_segy(file_path):
    """
    Проверяет, является ли SEGY файл 3D-сейсмограммой.

    :param file_path: Путь к SEGY файлу.
    :return: Строка с информацией о 2D/3D и размерности.
    """
    try:
        with segyio.open(file_path, "r", ignore_geometry=False) as f:
            # Проверяем наличие inline и crossline индексов
            num_inlines = len(f.ilines) if hasattr(f, "ilines") else None
            num_crosslines = len(f.xlines) if hasattr(f, "xlines") else None
            num_samples = len(f.samples)

            if num_inlines and num_crosslines:
                return f"Файл {os.path.basename(file_path)}: 3D-сейсмограмма. Размер: {num_inlines}×{num_crosslines}×{num_samples}"
            else:
                return f"Файл {os.path.basename(file_path)}: Возможно, 2D-сейсмограмма. Трасс: {len(f.trace)}, Сэмплов: {num_samples}"
    except Exception as e:
        return f"Файл {os.path.basename(file_path)}: Ошибка при обработке - {e}"

def analyze_segy_files(data_dir):
    """
    Анализирует файлы в указанной папке на предмет их 2D/3D структуры.

    :param data_dir: Директория с SEGY файлами.
    """
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.segy') or file_name.endswith('.sgy'):
            file_path = os.path.join(data_dir, file_name)
            result = check_if_3d_segy(file_path)
            print(result)

if __name__ == "__main__":
    # Укажите путь к папке с файлами
    data_dir = r"C:\Users\Daniil\PycharmProjects\SwinTransformer\data"
    analyze_segy_files(data_dir)
