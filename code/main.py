import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def TSNE_clustering(data_in: pd.DataFrame, do_plot=False, tsne_components=3):
    data = data_in.copy()
    components = 35

    X_data = data.drop(
        ["signal_id", "target_cluster", "target_point_1", "target_point_2", "target_point_3", "target_point_4", "X", "Y"]
        , axis=1
    )
    y_data = data["target_cluster"]
    pca = PCA(random_state=42, n_components=components)
    pca_data = pca.fit_transform(X_data)
    tsne = TSNE(random_state=42, perplexity=60, early_exaggeration=25, n_iter=1000, n_components=tsne_components)
    tsne_repr = tsne.fit_transform(pca_data)

    if do_plot:
        plt.figure(figsize=(15, 10))
        for label in np.unique(y_data):
            plt.scatter(tsne_repr[y_data == label, 0], tsne_repr[y_data == label, 1], label=label)
        plt.legend()
        plt.show()
    return tsne_repr


def k_neighbors_clustering(data_in: pd.DataFrame, do_plot: bool = False):
    data = data_in.copy()
    train_data = data[data["target_cluster"] != -1]
    test_data = data[data["target_cluster"] == -1]
    X_train, y_train = train_data.drop(
        ["signal_id", "target_cluster", "target_point_1", "target_point_2", "target_point_3", "target_point_4", "X",
         "Y"],
        axis=1
    ), train_data["target_cluster"]
    X_test = test_data.drop(
        ["signal_id", "target_cluster", "target_point_1", "target_point_2", "target_point_3", "target_point_4", "X",
         "Y"],
        axis=1
    )
    k_neigh = KNeighborsClassifier(n_neighbors=3)
    k_neigh.fit(X_train, y_train)
    y_hat_KN = k_neigh.predict(X_test)

    if do_plot:
        tsne_data = TSNE_clustering(test_data, do_plot=False, tsne_components=3)
        plt.figure(figsize=(15, 10))
        for label in np.unique(y_hat_KN):
            plt.scatter(tsne_data[y_hat_KN == label, 0], tsne_data[y_hat_KN == label, 1], label=label)
        plt.legend()
        plt.show()

    data.loc[data["target_cluster"] == -1, "target_cluster"] = y_hat_KN
    return data


def points_calculation(data_in: pd.DataFrame):
    data = data_in.copy()
    points_cols = [i for i in data.columns if "point" in i]
    data[points_cols] = data[points_cols].replace(-1, np.nan)

    for point in points_cols:
        data[point] = data.groupby(["target_cluster"], sort=False)[point].apply(lambda x: x.fillna(x.median()))
    return data


def main():
    PATH = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
    signals_data = pd.read_csv(PATH + "\\data\\signals.csv", header=None)
    signals_data = signals_data.rename(columns={0: "signal_id", 1: "X", 2: "Y"})
    dict_names = {}
    names_range = np.arange(1, 5001)

    for name in names_range:
        dict_names[name + 2] = f"V_{name}"

    dict_names[5003] = "target_cluster"
    dict_names[5004] = "target_point_1"
    dict_names[5005] = "target_point_2"
    dict_names[5006] = "target_point_3"
    dict_names[5007] = "target_point_4"

    signals_data = signals_data.rename(columns=dict_names)
    # #1 Task
    cluster_calculation = k_neighbors_clustering(signals_data)
    # #2 Task
    # Simplest solution :)
    points = points_calculation(cluster_calculation)
    points.to_csv(PATH + "\\data\\result.csv", header=None)

    print("Success")


if __name__ == "__main__":
    main()
