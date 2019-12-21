import tensorflow as tf
from sklearn.cluster import KMeans

@tf.function
def clustering_loss(inputs, centroids, alpha):
    inputs_expanded = inputs[tf.newaxis, :]
    centroids_expanded = centroids[:, tf.newaxis, :]
    distances = tf.reduce_mean(tf.math.squared_difference(inputs_expanded, centroids_expanded), 2)
    min_distances = tf.reduce_min(distances, 0)
    exp_distances = tf.exp(-alpha * (distances - min_distances))
    clust_loss = tf.reduce_mean(distances * exp_distances / tf.reduce_sum(exp_distances, axis=0))
    return clust_loss


class DeepKMeansAutoEncoder(tf.keras.Model):
    def __init__(self,
                 original_dim,
                 n_clusters=10,
                 lmbda=.1,
                 alpha=1000,
                 name='deep-k-means',
                 **kwargs):
        super(DeepKMeansAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.training = True
        # layers
        self.dense1 = tf.keras.layers.Dense(500, input_dim=original_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(500, activation="relu")
        self.dense3 = tf.keras.layers.Dense(2000, activation="relu")
        self.embedding = tf.keras.layers.Dense(n_clusters)
        self.dense4 = tf.keras.layers.Dense(500, activation="relu")
        self.dense5 = tf.keras.layers.Dense(500, activation="relu")
        self.dense6 = tf.keras.layers.Dense(2000, activation="relu")
        self.out = tf.keras.layers.Dense(original_dim)
        # clustering
        self.embeddings = None
        self.assignments = None
        self.centroids = tf.Variable(tf.zeros((n_clusters, n_clusters)), dtype=tf.float32, name="centroids",
                                     trainable=False)
        self.lmbda = lmbda
        self.n_clusters = n_clusters
        self.rec_loss_fn = tf.keras.losses.MeanSquaredError()
        self.clust_loss_fn = clustering_loss
        self.alpha = alpha

    def call(self, inputs, **kwargs):
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
            self.add_loss(self.lmbda * self.clust_loss_fn(z, self.centroids, alpha=self.alpha))
        return z

    def pre_train(self,
                  inputs,
                  epochs=50,
                  batch_size=64,
                  **kwargs):
        self.training = False
        self.fit(inputs, inputs, epochs=epochs, batch_size=batch_size, **kwargs)
        self.embeddings = self.predict(inputs)
        self.centroids.assign(KMeans(n_clusters=self.n_clusters).fit(self.embeddings).cluster_centers_)

    def train(self,
              inputs,
              epochs=10,
              batch_size=64,
              **kwargs):
        self.training = True
        for epoch in range(epochs):
            if kwargs['verbose'] != 0:
                print("Epoch %d/%d" % (epoch+1, epochs))
            self.fit(inputs, inputs, epochs=epochs, batch_size=batch_size, **kwargs)
            km = KMeans(n_clusters=self.n_clusters)
            self.embeddings = self.predict(inputs)
            km.fit(self.embeddings)
            self.assignments = km.labels_
            self.centroids.assign(km.cluster_centers_)
