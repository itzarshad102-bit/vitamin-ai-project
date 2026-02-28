import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "skin_dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=8,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "skin_dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=8,
    class_mode="binary",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN model...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

model.save("skin_model.h5")

print("CNN model saved successfully!")