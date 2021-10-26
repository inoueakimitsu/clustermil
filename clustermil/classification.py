from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import ClusterMixin, BaseEstimator, BaseEstimator, ClassifierMixin
import torch

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

def n_mis_predict_instance_per_bag(
        W, upper_threshold,
        lower_threshold,
        bag_cluster_table,
        l1_penalty_coef: float=1000):
    
    n_instance_under_threshold = torch.sum(
        (-1) * torch.minimum(
            lower_threshold - torch.matmul(bag_cluster_table, W),
            torch.zeros_like(lower_threshold)))

    n_instance_over_threshold = torch.sum(
        (-1) * torch.maximum(
            torch.matmul(bag_cluster_table, W) - upper_threshold,
            torch.zeros_like(upper_threshold)))
    
    n_bags = bag_cluster_table.shape[0]

    l1_penalty_term = torch.mean(torch.abs(W)) * l1_penalty_coef

    return (n_instance_under_threshold + n_instance_over_threshold) / n_bags + l1_penalty_term

def generate_mil_classifier(
        clustering: ClusterMixin,
        cluster_encoder: OneHotEncoder,
        bags: List[np.ndarray],
        lower_threshold: np.ndarray,
        upper_threshold: np.ndarray,
        n_clusters: int,
        n_epoch: int = 100,
        lr: int = 0.1,
        l1_penalty_coef: float = 1000,
        n_init: int = 100,
        debug: bool = False):
    
    one_hot_encoded = [
        cluster_encoder.transform(
            clustering.predict(b).reshape((-1, 1))) for b in bags]
    
    bag_cluster_table = np.array(
        [np.sum(x, axis=0) for x in one_hot_encoded])
    
    n_classes = lower_threshold.shape[1]

    # cast to torch tensor
    upper_threshold = torch.as_tensor(upper_threshold)
    lower_threshold = torch.as_tensor(lower_threshold)
    bag_cluster_table = torch.as_tensor(bag_cluster_table)

    estimated_W_list = []
    losses_list = []

    for i_init in range(n_init):
        print(i_init, "th init")

        W = torch.rand((n_clusters, n_classes), dtype=torch.float64, requires_grad=True)

        losses = []

        for i_epoch in range(n_epoch):
        
            loss = n_mis_predict_instance_per_bag(W, upper_threshold, lower_threshold, bag_cluster_table, l1_penalty_coef=l1_penalty_coef)
            loss.backward()

            with torch.no_grad():
                W -= lr * W.grad
                W.grad.zero_()
                W.data = torch.maximum(W.data, torch.zeros_like(W.data)).data
                eps = 1e-6
                W.data = W.data / (torch.sum(W.data, dim=0, keepdim=True)+eps)
                W.data = W.data / (torch.sum(W.data, dim=1, keepdim=True)+eps)
            
            losses.append(loss.item())

        # record this configuration
        estimated_W_list.append(W.detach().numpy())
        losses_list.append(losses)

    return ClusterMilClassifier(clustering, estimated_W_list, cluster_encoder)

