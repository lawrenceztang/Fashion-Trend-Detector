from clustering_utility import *
from utility import *


representation_learner = tf.keras.models.load_model('representation_learner', custom_objects={"Custom":RepresentationLearner})
encoder = representation_learner.encoder
data_preprocessing = representation_learner.data_preprocessing
data_augmentation = representation_learner.data_augmentation
x_data = get_data()
# data_preprocessing = tf.keras.models.load_model('data_preprocessing')
"""
## Compute the nearest neighbors
"""


"""
### Generate the embeddings for the images
"""

# Here batch size must be smaller again for memory. Not a problem because it is not training, simply predicting outputs
batch_size = 5
# Get the feature vector representations of the images.
feature_vectors = encoder.predict(data_preprocessing(x_data), batch_size=batch_size, verbose=1)
# Normalize the feature vectores.
feature_vectors = tf.math.l2_normalize(feature_vectors, -1)

"""
### Find the *k* nearest neighbours for each embedding
"""

neighbours = []
num_batches = feature_vectors.shape[0] // batch_size
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    current_batch = feature_vectors[start_idx:end_idx]
    # Compute the dot similarity.
    similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
    # Get the indices of most similar vectors.
    _, indices = tf.math.top_k(similarities, k=k_neighbours + 1, sorted=True)
    # Add the indices to the neighbours.
    neighbours.append(indices[..., 1:])

neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))

"""
Let's display some neighbors on each row
"""

nrows = 4
ncols = k_neighbours + 1

plt.figure(figsize=(12, 12))
position = 1
for _ in range(nrows):
    anchor_idx = np.random.choice(range(x_data.shape[0]))
    neighbour_indicies = neighbours[anchor_idx]
    indices = [anchor_idx] + neighbour_indicies.tolist()
    for j in range(ncols):
        plt.subplot(nrows, ncols, position)
        plt.imshow(x_data[indices[j]].astype("uint8"))
        plt.axis("off")
        position += 1
plt.show()
plt.savefig("neighbors_plot")

"""
### Train model
"""
import os

# If tune_encoder_during_clustering is set to False,
# then freeze the encoder weights.
for layer in encoder.layers:
    layer.trainable = tune_encoder_during_clustering
# Create the clustering model and learner.
clustering_model = create_clustering_model(encoder, num_clusters, data_preprocessing, data_augmentation, name="clustering")
clustering_learner = create_clustering_learner(clustering_model)
# Instantiate the model losses.
losses = [ClustersConsistencyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]
# Create the model inputs and labels.


labels = tf.ones(shape=(x_data.shape[0]))
# Compile the model.
clustering_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
    loss=losses,
)


inputs = {"anchors": x_data, "neighbours": tf.gather(x_data, neighbours)}
clustering_learner.fit(x=inputs, y=labels, batch_size=5, epochs=2)

gen = DataGenerator2(x_data, labels, neighbours, 1000)
history = clustering_learner.fit(gen, epochs=2)

"""
Plot training loss
"""
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
# plt.show()

"""
## Cluster analysis
"""

"""
### Assign images to clusters
"""

# Get the cluster probability distribution of the input images.
clustering_probs = clustering_model.predict(x_data, batch_size=batch_size, verbose=1)
# Get the cluster of the highest probability.
cluster_assignments = tf.math.argmax(clustering_probs, axis=-1).numpy()
# Store the clustering confidence.
# Images with the highest clustering confidence are considered the 'prototypes'
# of the clusters.
cluster_confidence = tf.math.reduce_max(clustering_probs, axis=-1).numpy()

"""
Let's compute the cluster sizes
"""

clusters = defaultdict(list)
for idx, c in enumerate(cluster_assignments):
    clusters[c].append((idx, cluster_confidence[idx]))

for c in range(num_clusters):
    print("cluster", c, ":", len(clusters[c]))

"""
Notice that the clusters have roughly balanced sizes.
"""

"""
### Visualize cluster images
Display the *prototypes*—instances with the highest clustering confidence—of each cluster:
"""

num_images = 8
plt.figure(figsize=(15, 15))
position = 1
for c in range(num_clusters):
    cluster_instances = sorted(clusters[c], key=lambda kv: kv[1], reverse=True)

    for j in range(min(num_images, len(cluster_instances))):
        image_idx = cluster_instances[j][0]
        plt.subplot(num_clusters, num_images, position)
        plt.imshow(x_data[image_idx].astype("uint8"))
        plt.axis("off")
        position += 1

plt.show()
plt.savefig("model_examples.png")
clustering_model.save("custering_model")

