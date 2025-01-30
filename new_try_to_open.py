import os
from obspy.io.segy.segy import _read_segy

def analyze_segy_with_obspy(file_path):
    """
Попытка проанализировать SEGY файл с помощью obspy.

:param file_path: Путь к SEGY файлу.
    """
    try:
        # Читаем файл с помощью obspy
        segy_data = _read_segy(file_path, headonly=True)
        print(f"Файл: {os.path.basename(file_path)}")

        # Общее количество трасс
        num_traces = len(segy_data.traces)
        print(f"Количество трасс: {num_traces}")

        # Длина трассы
        trace_lengths = [len(trace.data) for trace in segy_data.traces]
        unique_lengths = set(trace_lengths)

        if len(unique_lengths) == 1:
            print(f"Все трассы имеют одинаковую длину: {list(unique_lengths)[0]} сэмплов")
        else:
            print(f"Разные длины трасс обнаружены: {unique_lengths}")
            print(f"Минимальная длина трассы: {min(unique_lengths)}")
            print(f"Максимальная длина трассы: {max(unique_lengths)}")

        # Пример данных первой трассы
        first_trace = segy_data.traces[0].data
        print(f"Первые 10 значений первой трассы: {first_trace[:10]}")

    except Exception as e:
        print(f"Ошибка при обработке файла {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    # Укажите пути к файлам
    file_paths = [
        r"C:\Users\Daniil\PycharmProjects\SwinTransformer\data\7m_shots_0201_0329.segy",
        r"C:\Users\Daniil\PycharmProjects\SwinTransformer\data\1997_2.5D_shots.segy"
    ]

    for file_path in file_paths:
        print(f"Чтение файла: {file_path}")
        analyze_segy_with_obspy(file_path)
        print("\n" + "-" * 50 + "\n")
        