from alswr import ALS
import numpy as np
from sklearn.metrics import mean_squared_error


def load_csv(filename, delimiter="\t"):
    result = []
    with open(filename, "r") as src:
        for line in src:
            result.append(line.strip().split(delimiter))
    return result


def load_ratings(filename, delimiter="\t"):
    ratings = np.array(load_csv(filename, delimiter), dtype=int)[:, :3]
    ratings[:, 0] = ratings[:, 0] - 1
    ratings[:, 1] = ratings[:, 1] - 1
    return ratings


if __name__ == "__main__":
    n_user = 943
    n_item = 1682
    ratings = load_ratings("ml-100k/u1.base")
    # ratings = sp.csr_matrix((ratings[:, 2], (ratings[:, 1], ratings[:, 0])),
    #                         shape=(n_item, n_user))
    als = ALS(n_user, n_item, n_feature=50)
    als.fit(ratings)

    test_ratings = load_ratings("ml-100k/u1.test")
    print(np.max(test_ratings, axis=0))
    rmse = np.sqrt(mean_squared_error(y_pred=als.predict(test_ratings), y_true=test_ratings[:, 2]))
    print('rmse', rmse)
