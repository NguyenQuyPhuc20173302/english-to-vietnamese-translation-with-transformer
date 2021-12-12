from load_data import *
from model_transformer import model


if __name__=='__main__':
    transformer = model()
    transformer.summary()
    epochs = 1000
    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="accuracy", patience=early_stopping_patience, restore_best_weights=True
    )
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='model.hdf5',
        monitor='accuracy',
        save_best_only=True,
        save_weights_only=True,
    )

    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    transformer.load_weights('model.hdf5')
    transformer.fit(train_ds, epochs=100, callbacks=[early_stopping, Checkpoint])