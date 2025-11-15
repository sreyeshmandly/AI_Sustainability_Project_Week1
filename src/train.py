import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from preprocessing import create_data_generators


def build_model(input_shape=(224,224,3), num_classes=2):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def train():
    train_path = "../dataset/TRAIN"
    test_path  = "../dataset/TEST"

    train_gen, val_gen = create_data_generators(train_path, test_path)

    num_classes = len(train_gen.class_indices)

    model = build_model(num_classes=num_classes)

    model.compile(optimizer=Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    checkpoint = ModelCheckpoint("../models/waste_classifier.h5",
                                 monitor="val_accuracy",
                                 save_best_only=True)

    earlystop = EarlyStopping(monitor="val_loss", patience=5)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[checkpoint, earlystop]
    )

    # Save class labels
    with open("../models/class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    # Plot results
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()
    plt.savefig("../models/training_plot.png")


if __name__ == "__main__":
    train()
