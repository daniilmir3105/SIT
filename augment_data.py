import os
import numpy as np

def preprocess_seismic_data(seismic_data, noise_level=0.1):
    """
    Добавляет шум в сейсмические данные.

    Parameters:
        seismic_data (np.ndarray): Оригинальные сейсмические данные.
        noise_level (float): Уровень шума (доля от стандартного отклонения).

    Returns:
        np.ndarray: Сейсмические данные с добавленным шумом.
    """
    # Добавляем гауссов шум
    noise = np.random.normal(0, noise_level * np.std(seismic_data), seismic_data.shape)
    noisy_data = seismic_data + noise
    return noisy_data

def augment_data(input_dir, output_dir, noise_levels):
    """
    Выполняет аугментацию данных, добавляя шумы с разными уровнями.

    Parameters:
        input_dir (str): Путь к папке с исходными данными.
        output_dir (str): Путь к папке для сохранения аугментированных данных.
        noise_levels (list): Список уровней шума.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_dir, file_name)
            seismic_data = np.load(file_path)

            # Создаем 9 зашумленных копий с разными уровнями шума
            for i, noise_level in enumerate(noise_levels):
                augmented_data = preprocess_seismic_data(seismic_data, noise_level)
                augmented_file_name = f"{os.path.splitext(file_name)[0]}_noise_{i + 1}.npy"
                augmented_file_path = os.path.join(output_dir, augmented_file_name)
                np.save(augmented_file_path, augmented_data)

    # Подсчитываем количество файлов в выходной папке
    file_count = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    print(f"Всего файлов в папке '{output_dir}': {file_count}")

if __name__ == "__main__":
    # Определяем путь текущей директории, где находится исполняемый файл
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "data")
    output_dir = os.path.join(current_dir, "augmented_data")

    # Уровни шума (от 0.1 до 1.0 с шагом 0.1)
    noise_levels = [0.1 * i for i in range(1, 10)]

    # Выполняем аугментацию данных
    augment_data(input_dir, output_dir, noise_levels)
