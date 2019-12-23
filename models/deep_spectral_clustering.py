import scipy
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import numpy as np



class DeepSpectralClusteringAutoEncoder(tf.keras.Model):
    def __init__(self,
                 original_dim,
                 n_clusters=10,
                 lmbda=.1,
                 name='deep-spectral-clustering',
                 normalize=False,
                 n_neighbors=10,
                 **kwargs):
        super(DeepSpectralClusteringAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.training = True
        self.normalize = normalize
        # layers
        self.dense1 = tf.keras.layers.Dense(500, input_dim=original_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(500, activation="relu")
        self.dense3 = tf.keras.layers.Dense(2000, activation="relu")
        self.embedding = tf.keras.layers.Dense(n_clusters)
        self.dense4 = tf.keras.layers.Dense(2000, activation="relu")
        self.dense5 = tf.keras.layers.Dense(500, activation="relu")
        self.dense6 = tf.keras.layers.Dense(500, activation="relu")
        self.out = tf.keras.layers.Dense(original_dim)
        # clustering
        self.embeddings = None
        self.assignments = None
        self.centroids = np.zeros((n_clusters, n_clusters), dtype=np.float32)
        self.lmbda = lmbda
        self.n_clusters = n_clusters
        self.rec_loss_fn = tf.keras.losses.MeanSquaredError()
        self.B = None
        self.n_neighbors = n_neighbors

    def call(self, inputs, **kwargs):
        if self.training:
            B = inputs[:, :self.n_clusters]
            inputs = inputs[:, self.n_clusters:]
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        z = self.embedding(x)
        x = self.dense4(z)
        x = self.dense5(x)
        x = self.dense6(x)
        reconstructed = self.out(x)
        self.add_loss(self.rec_loss_fn(reconstructed, inputs))
        if self.training:
            self.add_loss(self.lmbda * self.rec_loss_fn(z, B))
            # return B
        return z

    def pre_train(self,
                  inputs,
                  epochs=50,
                  batch_size=64,
                  **kwargs):
        self.training = False
        self.fit(inputs, inputs, epochs=epochs, batch_size=batch_size, **kwargs)

    def train(self,
              inputs,
              epochs=10,
              batch_size=64,
              **kwargs):
        self.embeddings = self.predict(inputs)
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(self.embeddings)
        self.assignments = km.labels_
        self.centroids = km.cluster_centers_
        self.spectral_update()
        self.training = True
        for epoch in range(epochs):
            if kwargs['verbose'] != 0:
                print("Epoch %d/%d" % (epoch+1, epochs))
            self.fit(np.hstack([self.B, inputs]), inputs, epochs=1, batch_size=batch_size, **kwargs)
            self.embeddings = self.predict(inputs)
            self.spectral_update()

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        self.training = False
        predictions = super(DeepSpectralClusteringAutoEncoder, self).predict(x, batch_size, verbose, steps, callbacks,
                                                                             max_queue_size, workers,
                                                                             use_multiprocessing)
        self.training = True
        return predictions

    def spectral_update(self):
        G = np.eye(self.centroids.shape[0])[self.assignments]
        dist_matrix = kneighbors_graph(self.embeddings, n_neighbors=self.n_neighbors, mode="distance")
        m = np.mean(dist_matrix.data)
        v = np.var(self.embeddings)
        normalized_data = np.exp(-((dist_matrix.data - m) ** 2) / (2 * v))
        W = csr_matrix((normalized_data, dist_matrix.indices, dist_matrix.indptr), shape=dist_matrix.shape)
        if self.normalize:
            D_12 = scipy.sparse.diags(np.asarray(W.sum(axis=1)).ravel() ** (-1 / 2))
            W = D_12 @ W @ D_12
        U, _, Vt = scipy.linalg.svd(W @ G @ self.centroids + self.lmbda * self.embeddings,
                                    full_matrices=False)
        self.B = U @ Vt
        WB = W @ self.B
        self.centroids = scipy.linalg.pinv(G.T @ G) @ G.T @ WB
        self.assignments = np.argmin(np.sum((WB - self.centroids[:, np.newaxis]) ** 2, axis=2), axis=0)
