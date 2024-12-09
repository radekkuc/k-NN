import csv
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def str_column_to_float(dataset, column):
    for row in dataset:
        if is_float(row[column]):
            row[column] = float(row[column])
        else:
            print(f"Warning: Non-numeric value '{row[column]}' in column {column}")
            row[column] = 0.0

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = sorted(set(class_values))
    lookup = {value: i for i, value in enumerate(unique)}
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def load_csv_dynamic(filename, separator=",", class_column=None):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=separator)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    if class_column is None:
        class_column = len(dataset[0]) - 1
    return dataset, class_column

def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        minmax.append([min(col_values), max(col_values)])
    return minmax

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            min_val, max_val = minmax[i]
            denominator = max_val - min_val
            row[i] = (row[i] - min_val) / denominator if denominator != 0 else 0.0

def euclidean_distance(row1, row2):
    return sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1)-1)))

def get_neighbors(train, test_row, num_neighbors):
    distances = [(train_row, euclidean_distance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])
    num_neighbors = min(num_neighbors, len(distances))
    return [distances[i][0] for i in range(num_neighbors)]

def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def calculate_accuracy(train, test, num_neighbors):
    correct = 0
    for row in test:
        prediction = predict_classification(train, row, num_neighbors)
        if prediction == row[-1]:
            correct += 1
    return correct / len(test) * 100

if __name__ == '__main__':
    filename = 'bezdekIris.data.'  # Plie z danymi
    separator = ','
    class_column = None

    # Wczytaj zbiór danych
    dataset, class_column = load_csv_dynamic(filename, separator, class_column)

    # Konwersja wartości numerycznych do float
    for i in range(len(dataset[0])):
        if i != class_column:
            str_column_to_float(dataset, i)

    # Konwersja klas do liczb całkowitych
    lookup = str_column_to_int(dataset, class_column)
    print(f"Class Mapping: {lookup}")

    # Normalizacja cech
    features = [row[:class_column] + row[class_column + 1:] for row in dataset]
    minmax = dataset_minmax(features)
    normalize_dataset(features, minmax)

    # Podział danych na zbiór treningowy i testowy
    train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=[row[class_column] for row in dataset])

    # Algorytm k-NN from scratch
    num_neighbors = 3
    scratch_accuracy = calculate_accuracy(train_data, test_data, num_neighbors)
    print(f"Accuracy (from scratch): {scratch_accuracy:.2f}%")

    # Przygotowanie danych dla scikit-learn
    X_train = [row[:-1] for row in train_data]  # Cechy zbioru treningowego
    y_train = [row[-1] for row in train_data]  # Klasy zbioru treningowego
    X_test = [row[:-1] for row in test_data]  # Cechy zbioru testowego
    y_test = [row[-1] for row in test_data]  # Klasy zbioru testowego

    # Algorytm k-NN z scikit-learn
    model = KNeighborsClassifier(n_neighbors=num_neighbors)
    model.fit(X_train, y_train)  # Trening modelu
    sklearn_accuracy = model.score(X_test, y_test) * 100  # Obliczenie dokładności
    print(f"Accuracy (scikit-learn): {sklearn_accuracy:.2f}%")
