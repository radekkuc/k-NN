import unittest
from knn import (
    load_csv_dynamic,
    str_column_to_float,
    str_column_to_int,
    dataset_minmax,
    normalize_dataset,
    euclidean_distance,
    get_neighbors,
    predict_classification,
    calculate_accuracy
)

class TestKNN(unittest.TestCase):
    def setUp(self):
        # Przykładowy zbiór danych do testów
        self.dataset = [
            [5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
            [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
            [7.0, 3.2, 4.7, 1.4, 'Iris-versicolor'],
            [6.4, 3.2, 4.5, 1.5, 'Iris-versicolor'],
            [6.3, 3.3, 6.0, 2.5, 'Iris-virginica']
        ]
        self.class_column = len(self.dataset[0]) - 1  # Ostatnia kolumna to klasy

    def test_load_csv_dynamic(self):
        # Testowanie wczytywania pliku CSV
        dataset, class_column = load_csv_dynamic('bezdekIris.data', ',', None)
        self.assertGreater(len(dataset), 0)
        self.assertEqual(class_column, len(dataset[0]) - 1)

    def test_str_column_to_float(self):
        # Test konwersji cech na float
        str_column_to_float(self.dataset, 0)
        str_column_to_float(self.dataset, 1)
        for row in self.dataset:
            self.assertIsInstance(row[0], float)
            self.assertIsInstance(row[1], float)

    def test_str_column_to_int(self):
        # Test konwersji klas do liczb całkowitych
        lookup = str_column_to_int(self.dataset, self.class_column)
        self.assertEqual(len(lookup), 3)  # Powinny być 3 unikalne klasy
        for row in self.dataset:
            self.assertIsInstance(row[self.class_column], int)

    def test_dataset_minmax(self):
        # Test min i max w cechach
        str_column_to_float(self.dataset, 0)
        minmax = dataset_minmax(self.dataset)
        self.assertEqual(len(minmax), self.class_column)
        for col_minmax in minmax:
            self.assertLessEqual(col_minmax[0], col_minmax[1])

    def test_normalize_dataset(self):
        # Test normalizacji danych
        str_column_to_float(self.dataset, 0)
        minmax = dataset_minmax(self.dataset)
        normalize_dataset(self.dataset, minmax)
        for row in self.dataset:
            for i in range(self.class_column):
                self.assertGreaterEqual(row[i], 0.0)
                self.assertLessEqual(row[i], 1.0)

    def test_euclidean_distance(self):
        # Test odległości euklidesowej
        distance = euclidean_distance([0, 0, 0], [3, 4, 0])
        self.assertEqual(distance, 5.0)

    def test_get_neighbors(self):
        # Test znajdowania sąsiadów
        str_column_to_float(self.dataset, 0)
        neighbors = get_neighbors(self.dataset, self.dataset[0], 3)
        self.assertEqual(len(neighbors), 3)

    def test_predict_classification(self):
        # Test predykcji klasy
        str_column_to_float(self.dataset, 0)
        str_column_to_int(self.dataset, self.class_column)
        prediction = predict_classification(self.dataset, self.dataset[0], 3)
        self.assertIsInstance(prediction, int)

    def test_calculate_accuracy(self):
        # Test dokładności
        str_column_to_float(self.dataset, 0)
        str_column_to_int(self.dataset, self.class_column)
        accuracy = calculate_accuracy(self.dataset, self.dataset, 3)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 100.0)

if __name__ == '__main__':
    unittest.main()
