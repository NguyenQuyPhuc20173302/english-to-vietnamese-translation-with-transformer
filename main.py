from load_data import *
from model_transformer import model


if __name__=='__main__':
    transformer = model()
    transformer.summary()

    Checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='model.hdf5',
        monitor='val_loss',
        save_best_only=True)

    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    transformer.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[Checkpoint])