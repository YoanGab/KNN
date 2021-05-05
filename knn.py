# %% Import
import numpy as np
import pandas as pd

# %% KNN Class
class KNN:
    def __init__(self, k=3, X=None, Y=None):
        self.k = k
        self.X_train = X
        self.y_train = Y

    def predict(self, X, distance):
        predicted_labels = [self.predict_one(x, distance) for x in X]
        return np.array(predicted_labels)

    def predict_one(self, x, distance):
        distances = [distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        dict_of_labels_weighted = {}
        for i, k_nearest_label in enumerate(k_nearest_labels):
            weight = 1/distances[k_indices[i]
                                 ]**2 if distances[k_indices[i]] != 0 else np.inf
            if k_nearest_labels[i] in dict_of_labels_weighted:
                dict_of_labels_weighted[k_nearest_label] += weight
            else:
                dict_of_labels_weighted[k_nearest_label] = weight
        dict_of_labels_weighted = sorted(
            dict_of_labels_weighted.items(), key=lambda item: item[1], reverse=True)
        return dict_of_labels_weighted[0][0]

# %% Distance calculus
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x2-x1))

def euclidean_distance(x1, x2):
    return np.sqrt(np.dot(x1 - x2, x1 - x2))

def cosine_distance(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1)*np.linalg.norm(x2))

# %% Confusion matrix
def confusion_matrix(y_true, y_pred):
    labels = sorted(set(y_true))
    confusion_matrix = np.zeros([len(labels), len(labels)])
    for index, val in enumerate(y_true):
        confusion_matrix[labels.index(y_pred[index])][labels.index(val)] += 1
    return labels, confusion_matrix

def display_confusion_matrix(confusion_matrix, labels):
    print("\t", "\t".join(labels))
    for index, val in enumerate(confusion_matrix):
        print(labels[index], "\t", "\t".join([str(int(v)) for v in val]))

# %% Data operation
def preprocessing_data(df, has_labels):
    max_index = range(df.shape[1] - 1) if has_labels else range(df.shape[1])
    for col_index in max_index:
        mean_col = np.mean(df.iloc[:, col_index])
        sd_col = np.std(df.iloc[:, col_index])
        df.iloc[:, col_index] = (df.iloc[:, col_index] - mean_col) / sd_col
    return df

def process_data(filename, has_labels=True):
    return preprocessing_data(pd.read_csv(filename, header=None), has_labels=has_labels)

def shuffle_data(df):
    return df.sample(frac=1)

def split_sets(df, batch_size):
    return df.iloc[:int(len(df)*batch_size)], df.iloc[int(len(df)*batch_size):]

def get_train_sets(df):
    return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()

def get_test_sets(df, has_labels=True):
    if has_labels:
        return get_train_sets(df)
    else:
        return df.to_numpy(), None

def train_model(knn, X_test, y_test, distance):
    predictions = knn.predict(X_test, distance)
    if y_test is not None and len(y_test) > 0:
        labels, df_confusion_matrix = confusion_matrix(predictions, y_test)
        display_confusion_matrix(df_confusion_matrix, labels)
        print("\n{:.2f}% de précision".format(
            sum(predictions == y_test) / len(predictions) * 100))

    unique, counts = np.unique(predictions, return_counts=True)
    print(dict(zip(unique, counts)))
    return predictions

def train_set(filename, batch_size, k, distance=euclidean_distance):
    df = shuffle_data(process_data(filename))
    train_df, test_df = split_sets(df, batch_size)
    X_train, y_train = get_train_sets(train_df)
    X_test, y_test = get_test_sets(test_df)
    knn = KNN(k, X_train, y_train)
    train_model(knn, X_test, y_test, distance)
    return knn

def predict_set(knn, filename, has_labels=True, distance=euclidean_distance):
    df = process_data(filename, has_labels)
    X_test, y_test = get_test_sets(df, has_labels)
    return train_model(knn, X_test, y_test, distance)

# %% Main
if __name__ == '__main__':
    data_folder = "./data"
    train_data = data_folder + "/data.csv"
    validation_data = data_folder + "/preTest.csv"
    to_predict_data = data_folder + "/finalTest.csv"

    # Train test
    print("Train test")
    knn = train_set(filename=train_data, batch_size=0.7,
                    k=15, distance=euclidean_distance)
    # Validation Test
    print("\nValidation test")
    predict_set(knn=knn, filename=validation_data, distance=euclidean_distance)

    # Final test
    print("\nfinal test")
    predictions = predict_set(
        knn=knn, filename=to_predict_data, has_labels=False, distance=euclidean_distance)

    with open("./output/gabison_yoan.txt", "w") as f:
        f.write("\n".join(predictions))