# clustermil

[![Build Status](https://app.travis-ci.com/inoueakimitsu/clustermil.svg?branch=main)](https://app.travis-ci.com/inoueakimitsu/clustermil)
<a href="https://github.com/inoueakimitsu/clustermil/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/inoueakimitsu/clustermil"></a> 

Python package for multiple instance learning (MIL) for large n_instance dataset.

## Features

- support count-based multiple instance assumptions (see [wikipedia](https://en.wikipedia.org/wiki/Multiple_instance_learning#:~:text=Presence-%2C%20threshold-%2C%20and%20count-based%20assumptions%5Bedit%5D))
- support multi-class setting
- support scikit-learn Clustering algorithms (such as `MiniBatchKMeans`)
- fast even if n_instance is large

## Installation

```bash
pip install clustermil
```

## Usage

```python
# Prepare follwing dataset
#
# - bags ... list of np.ndarray
#            (num_instance_in_the_bag * num_features)
# - lower_threshold ... np.ndarray (num_bags * num_classes)
# - upper_threshold ... np.ndarray (num_bags * num_classes)
#
# bags[i_bag] contains not less than lower_thrshold[i_bag, i_class]
# i_class instances.

# Prepare single-instance clustering algorithms
from sklearn.cluster import MiniBatchKMeans
n_clusters = 100
clustering = MiniBatchKMeans(n_clusters=n_clusters)
clusters = clustering.fit_predict(np.vstack(bags)) # flatten bags into instances

# Prepare one-hot encoder
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(clusters)

# generate ClusterMilClassifier with helper function
from clustermil import generate_mil_classifier

milclassifier = generate_mil_classifier(
            clustering,
            onehot_encoder,
            bags,
            lower_threshold,
            upper_threshold,
            n_clusters)

# after multiple instance learning,
# you can predict instance class
milclassifier.predict([instance_feature])
```

See `tests/test_classification.py` for an example of a fully working test data generation process.

## License

clustermil is available under the MIT License.
