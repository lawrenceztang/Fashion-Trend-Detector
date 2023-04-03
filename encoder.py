from utility import *


"""
## Self-supervised representation learning
"""


"""
### Train the model
"""

x_data = get_data()
x_gen = DataGenerator(x_data, 32)

mirrored_strategy = tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.get_strategy()

with mirrored_strategy.scope():
    # Create vision encoder.

    representation_learner = None
    try:
        representation_learner_loaded = tf.keras.models.load_model('representation_learner')
        representation_learner = RepresentationLearner(
            representation_learner_loaded.encoder, representation_learner_loaded.data_preprocessing,
            representation_learner_loaded.data_augmentation, projection_units, num_augmentations=2, temperature=0.1
        )
        representation_learner.projector = representation_learner_loaded.projector
    except:
        encoder = create_encoder(representation_dim)

        data_preprocessing = keras.Sequential(
            [
                layers.Resizing(target_size, target_size),
                layers.Normalization(),
            ]
        )

        data_augmentation = keras.Sequential(
            [
                layers.RandomTranslation(
                    height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
                ),
                layers.RandomFlip(mode="horizontal"),
                layers.RandomRotation(factor=0.15, fill_mode="nearest"),
                layers.RandomZoom(
                    height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
                ),
            ]
        )

        data_preprocessing.layers[-1].adapt(x_data)
        representation_learner = RepresentationLearner(
            encoder, data_preprocessing, data_augmentation, projection_units, num_augmentations=2, temperature=0.1
        )

    # Create a a Cosine decay learning rate scheduler.
    lr_scheduler = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=70, alpha=0.1
    )


# Compile the model.
representation_learner.compile(

    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
)


history = representation_learner.fit(
    x=x_gen,
    epochs=70,
    workers=8,
    use_multiprocessing=True
)
representation_learner.save('representation_learner')
"""
Plot training loss
"""

plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
plt.savefig("loss_graph_encoder")
