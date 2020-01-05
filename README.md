A comparaison of these deep clustering methods using *tensorflow2*: 
1. [Sequential AutoEncoder and $k$-Means](https://github.com/chakib401/deep-clustering/blob/master/models/seq_ae_km.py) where we  use an autoencoder to reduce the dimension of the data and then apply a k-Means Clustering to have our classes.
2. [Deep $k$-Means](https://github.com/chakib401/deep-clustering/blob/master/models/deep_km.py) which is an implementation of the method described in this [paper](https://arxiv.org/abs/1806.10069).
3. [Deep Spectral Clustering](https://github.com/chakib401/deep-clustering/blob/master/models/deep_spectral_clustering.py) a hybrid of an autoencoder and spectral clustering.
