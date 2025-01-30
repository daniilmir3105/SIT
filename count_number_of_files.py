import os

def count_files_in_data_folder():
    """
    Считает количество файлов в папке 'data', расположенной рядом с исполняемым файлом.

    Returns:
        int: Количество файлов в папке 'data'.
    """
    # Определяем текущую директорию, где находится исполняемый файл
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    # Проверяем, существует ли папка 'data'
    if not os.path.exists(data_dir):
        print(f"Папка '{data_dir}' не существует.")
        return 0

    # Подсчитываем количество файлов в папке 'data'
    file_count = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
    return file_count

if __name__ == "__main__":
    file_count = count_files_in_data_folder()
    print(f"Количество файлов в папке 'data': {file_count}")
