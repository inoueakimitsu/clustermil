import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder

from clustermil.classification import generate_mil_classifier

def generate_class_ratios(n_classes):
    
    while True:
        use_classes = np.random.choice([0, 1], size=n_classes, p=[.8, .2])
        if not np.all(use_classes == 0):
            break
    
    ratio_classes = use_classes * np.random.uniform(low=0, high=1, size=n_classes)
    ratio_classes = ratio_classes**4
    ratio_classes = ratio_classes / np.sum(ratio_classes)

    return ratio_classes

def generate_instance(n_classes, n_instances_of_each_bags):

    class_labels_of_intance_in_bags = [np.random.choice(
        np.arange(n_classes),
        size=n_instance_in_bag,
        p=generate_class_ratios(n_classes)) for n_instance_in_bag in n_instances_of_each_bags]
    
    return class_labels_of_intance_in_bags

class TestClassification:

    def test_fit(self):

        np.random.seed(123)

        n_classes = 15
        n_bags = 200
        n_max_instance_in_one_bag = 1000
        n_instances_of_each_bags = [np.random.randint(low=0, high=n_max_instance_in_one_bag) for _ in range(n_bags)]
        class_labels_of_instance_in_bags = generate_instance(n_classes, n_instances_of_each_bags)
        count_each_class_of_instance_in_bags = [
            pd.Series(x).value_counts().to_dict() for x in class_labels_of_instance_in_bags
        ]
        count_each_class_of_instance_in_bags_matrix = \
            pd.DataFrame(count_each_class_of_instance_in_bags)[list(range(n_classes))].values
        count_each_class_of_instance_in_bags_matrix = np.nan_to_num(count_each_class_of_instance_in_bags_matrix)
        lower_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        upper_threshold = np.zeros_like(count_each_class_of_instance_in_bags_matrix)
        divisions = [0, 50, 100, 200, 1000, n_max_instance_in_one_bag]
        for i_bag in range(n_bags):
            for i_class in range(n_classes):
                positive_count = count_each_class_of_instance_in_bags_matrix[i_bag, i_class]
                for i_division in range(len(divisions)-1):
                    if divisions[i_division] <= positive_count and positive_count < divisions[i_division+1]:
                        lower_threshold[i_bag, i_class] = divisions[i_division]
                        upper_threshold[i_bag, i_class] = divisions[i_division+1]

        n_fatures = 7
        x_min = 0
        x_max = 100
        cov_diag = 0.1*40**2

        means_of_classes = [np.random.uniform(low=x_min, high=x_max, size=n_fatures) for _ in range(n_classes)]
        covs_of_classes = [np.eye(n_fatures)*cov_diag for _ in range(n_classes)]
        bags = [
            np.vstack([
                np.random.multivariate_normal(
                    means_of_classes[class_label],
                    covs_of_classes[class_label],
                    size=1) for class_label in class_labels_of_instance_in_bag
            ]) for class_labels_of_instance_in_bag in class_labels_of_instance_in_bags
        ]

        true_y = [np.array([class_label for class_label in class_labels_of_instance_in_bag]) for class_labels_of_instance_in_bag in class_labels_of_instance_in_bags]

        # Show dataset structures

        print("len(bags) =", len(bags))
        print("bags[0].shape =", bags[0].shape)
        print("bags[0][:4, :5] =\n", bags[0][:4, :5])
        print("lower_threshold.shape = ", lower_threshold.shape)
        print("lower_threshold[:3, :4] = \n", lower_threshold[:3, :4])
        print("upper_threshold.shape = ", upper_threshold.shape)
        print("upper_threshold[:3, :4] = \n", upper_threshold[:3, :4])

        
        flatten_features = np.vstack(bags)
        max_n_clusters = 500
        # cluster_generator = KMeans(n_clusters=max_n_clusters, random_state=0)
        # cluster_generator = DBSCAN()# KMeans(n_clusters=n_clusters, random_state=0)
        cluster_generator = MiniBatchKMeans(n_clusters=max_n_clusters, random_state=0)
        insample_estimated_clusters = cluster_generator.fit_predict(flatten_features)
        n_clusters = np.max(insample_estimated_clusters) + 1
        print("n_clusters:", n_clusters)

        cluster_encoder = OneHotEncoder(sparse=False)
        cluster_encoder.fit(np.array([np.arange(n_clusters)]).T)

        milclassifier = generate_mil_classifier(
            cluster_generator,
            cluster_encoder,
            bags,
            lower_threshold,
            upper_threshold,
            n_clusters,
            n_epoch = 100,
            lr = 0.1,
            l1_penalty_coef = 1000,
            n_init = 10)

        df_confusion_matrix = pd.crosstab(np.hstack(true_y), milclassifier.predict(np.vstack(bags)))
        
        print("confusion matrix")
        print(df_confusion_matrix)

