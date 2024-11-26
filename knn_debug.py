import csv
from math import sqrt
from sklearn.model_selection import train_test_split

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if is_float(row[column]):
            row[column] = float(row[column])
        else:
            print(f"Warning: Non-numeric value '{row[column]}' in column {column}")
            row[column] = 0.0  # Wartość domyślna


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = sorted(set(class_values))
    lookup = {value: i for i, value in enumerate(unique)}
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Load CSV data
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:  # Pomijamy puste wiersze
                continue
            dataset.append(row)
    return dataset

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = []
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        minmax.append([min(col_values), max(col_values)])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            min_val, max_val = minmax[i]
            denominator = max_val - min_val
            row[i] = (row[i] - min_val) / denominator if denominator != 0 else 0.0

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    return sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(len(row1)-1)))

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = [(train_row, euclidean_distance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])
    num_neighbors = min(num_neighbors, len(distances))
    return [distances[i][0] for i in range(num_neighbors)]

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Calculate accuracy
def calculate_accuracy(train, test, num_neighbors):
    correct = 0
    for row in test:
        prediction = predict_classification(train, row, num_neighbors)
        if prediction == row[-1]:
            correct += 1
        print(f"Expected: {row[-1]}, Predicted: {prediction}")
    return correct / len(test) * 100

def load_csv_dynamic(filename, separator=",", class_column=None):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=separator)
        for row in csv_reader:
            if not row:  # Pomijamy puste wiersze
                continue
            dataset.append(row)

    # Automatyczne określenie kolumny klas (ostatnia, jeśli nie podano)
    if class_column is None:
        class_column = len(dataset[0]) - 1

    return dataset, class_column


if __name__ == '__main__':
    filename = 'bezdekIris.data'  
    separator = ','
    class_column = None

    # Wczytaj zbiór danych
    dataset, class_column = load_csv_dynamic(filename, separator, class_column)

    # Ustaw domyślną kolumnę klas, jeśli nie podano
    if class_column is None:
        class_column = len(dataset[0]) - 1  # Domyślnie ostatnia kolumna
    class_column = int(class_column)

    # Debug: Wyświetlenie kilku pierwszych wierszy po wczytaniu
    print("First 5 rows of dataset:")
    for row in dataset[:5]:
        print(row)

    # Konwersja wartości numerycznych do float (wszystkie poza kolumną klas)
    for i in range(len(dataset[0])):
        if i != class_column:
            str_column_to_float(dataset, i)

    # Konwersja klas do liczb całkowitych
    lookup = str_column_to_int(dataset, class_column)
    print(f"Class Mapping: {lookup}")

    # Normalizacja cech (wszystkie poza kolumną klas)
    features = [row[:class_column] + row[class_column + 1:] for row in dataset]
    minmax = dataset_minmax(features)

    for row in dataset:
        for i in range(len(row)):
            if i == class_column:
                continue  # Pomijamy kolumnę klas
            feature_index = i if i < class_column else i - 1
            if feature_index < len(minmax):
                min_val, max_val = minmax[feature_index]
                denominator = max_val - min_val
                row[i] = (row[i] - min_val) / denominator if denominator != 0 else 0.0

    # Debugging: Wyświetlenie znormalizowanych danych
    print("Normalized Dataset (First 10 rows):")
    for row in dataset[:10]:
        print(row)

    # Podział danych na zbiór treningowy i testowy
    train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=[row[class_column] for row in dataset])

    # Debugging: Wyświetlenie liczności zbiorów
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

    num_neighbors = 3

    # Obliczenie i wyświetlenie dokładności modelu
    try:
        accuracy = calculate_accuracy(train_data, test_data, num_neighbors)
        print(f"Model Accuracy: {accuracy:.2f}%")
    except IndexError as e:
        print(f"Error during accuracy calculation: {e}")





