from common import *

"""
Display a sample of augmented versions of the image
"""

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     augmented_images = data_augmentation(np.array([image]))
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")
# plt.show()

"""
## Self-supervised representation learning
"""

"""
### Implement the vision encoder
"""


def create_encoder(representation_dim):
    encoder = keras.Sequential(
        [
            keras.applications.ResNet50V2(
                include_top=False, weights=None, pooling="avg"
            ),
            layers.Dense(representation_dim),
        ]
    )
    return encoder


"""
### Implement the unsupervised contrastive loss
"""


class RepresentationLearner(keras.Model):
    def __init__(
            self,
            encoder,
            projection_units,
            num_augmentations,
            temperature=1.0,
            dropout_rate=0.1,
            l2_normalize=False,
            **kwargs
    ):
        super(RepresentationLearner, self).__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
                tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
                / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.math.reduce_max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


"""
### Train the model
"""

mirrored_strategy = tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.get_strategy()

with mirrored_strategy.scope():
    # Create vision encoder.
    encoder = create_encoder(representation_dim)

    #
    # Create representation learner.
    representation_learner = RepresentationLearner(
        encoder, projection_units, num_augmentations=2, temperature=0.1
    )
    # Create a a Cosine decay learning rate scheduler.
    lr_scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=70, alpha=0.1
    )

# Compile the model.
representation_learner.compile(

    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
)

representation_learner.encoder = tf.keras.models.load_model('encoder_model')
representation_learner.projector = tf.keras.models.load_model('projector_model')

history = representation_learner.fit(
    x=x_gen,
    epochs=70,
    workers=8,
    use_multiprocessing=True
)
representation_learner.encoder.save('encoder_model')
representation_learner.projector.save('projector_model')
"""
Plot training loss
"""

plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
plt.savefig("loss_graph_encoder")
