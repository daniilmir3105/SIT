import os


def count_files_in_directory(directory):
    """
    Считает количество файлов в указанной папке.

    :param directory: Путь к папке.
    :return: Количество файлов.
    """
    if not os.path.exists(directory):
        print(f"Папка {directory} не существует.")
        return 0

    file_count = sum(1 for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)))
    return file_count


if __name__ == "__main__":
    # Путь к папке с патчами
    directory = r"C:\Users\Daniil\PycharmProjects\SwinTransformer\patches"

    # Подсчет файлов
    file_count = count_files_in_directory(directory)
    print(f"Количество файлов в папке '{directory}': {file_count}")
