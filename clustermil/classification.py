from time import time
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import ClusterMixin, BaseEstimator, BaseEstimator, ClassifierMixin
import pulp

class ClusterMilClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clustering: ClusterMixin, weight_list: np.ndarray, onehot_encoder: OneHotEncoder):
        assert len(weight_list) > 0

        self.weight_list = weight_list
        self.clustering = clustering
        self.onehot_encoder = onehot_encoder
    def fit(self, X, y):
        # do nothing
        return self
    def predict(self, X):
        # bag_cluster_table (bag * cluster) @ weight (cluster*class)        
        new_cluster = self.clustering.predict(X)
        new_onehot_encoded_class = self.onehot_encoder.transform(np.array([new_cluster]).T)
        new_onehot_encoded_class.shape
        
        # its shape is (n_instance * n_class)
        n_instance = len(X)
        n_class = self.weight_list[0].shape[1]
        estimated_class_scores_sum = np.zeros((n_instance, n_class), dtype="float64")
        for i_W, W in enumerate(self.weight_list):
            estimated_class_scores = new_onehot_encoded_class @ W
            estimated_class_scores_sum += estimated_class_scores

        pred_y = np.argmax(estimated_class_scores_sum, axis=1)
        return pred_y


def generate_mil_classifier(
        clustering: ClusterMixin,
        cluster_encoder: OneHotEncoder,
        bags: List[np.ndarray],
        lower_threshold: np.ndarray,
        upper_threshold: np.ndarray,
        n_clusters: int,
        debug: bool = False):
    
    one_hot_encoded = [
        cluster_encoder.transform(
            clustering.predict(b).reshape((-1, 1))) for b in bags]
    
    bag_cluster_table = np.array(
        [np.sum(x, axis=0) for x in one_hot_encoded])
    
    n_classes = lower_threshold.shape[1]

    problem = pulp.LpProblem("clustermil", pulp.LpMinimize)
    margin = pulp.LpVariable("margin", lowBound=0)
    problem += margin

    W = []
    for i_class in range(n_classes):
        w = []
        for i_cluster in range(n_clusters):
            w.append(pulp.LpVariable(
                f"w_{i_class}_{i_cluster}", cat=pulp.LpBinary))
        W.append(w)
    for i_cluster in range(n_clusters):
        problem += pulp.lpSum(W[i_class][i_cluster] for i_class in range(n_classes)) == 1
    
    for i_bag in range(len(bags)):
        for i_class in range(n_classes):
            sum = None
            for i_cluster in range(n_clusters):
                if sum is None:
                    sum = W[i_class][i_cluster] * bag_cluster_table[i_bag][i_cluster]
                else:
                    sum += W[i_class][i_cluster] * bag_cluster_table[i_bag][i_cluster]
            
            problem += sum - margin <= upper_threshold[i_bag][i_class]
            problem += sum + margin >= lower_threshold[i_bag][i_class]
    
    
    problem.solve()
    W_float = np.array([np.array([x.value() for x in w]) for w in W]).T
    estimated_W_list = [W_float]

    if debug:
        print(f"W_float: {W_float}")
        print(f"margin: {margin.value()}")

    return ClusterMilClassifier(clustering, estimated_W_list, cluster_encoder)

