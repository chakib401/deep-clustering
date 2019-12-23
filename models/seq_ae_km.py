import tensorflow as tf
from sklearn.cluster import KMeans


class SeqAutoEncoderKMeans(tf.keras.Model):
    def __init__(self,
                 original_dim,
                 n_clusters=10,
                 name='seq-ae-k-means',
                 **kwargs):
        super(SeqAutoEncoderKMeans, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
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
        self.centroids = tf.zeros((n_clusters, n_clusters))
        self.n_clusters = n_clusters
        self.rec_loss_fn = tf.keras.losses.MeanSquaredError()

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
        return z

    def train(self,
              inputs,
              epochs=50,
              batch_size=64,
              **kwargs):
        self.fit(inputs, inputs, epochs=epochs, batch_size=batch_size, **kwargs)
        km = KMeans(n_clusters=self.n_clusters)
        self.embeddings = self.predict(inputs)
        km.fit(self.embeddings)
        self.assignments = km.labels_
        self.centroids = km.cluster_centers_
