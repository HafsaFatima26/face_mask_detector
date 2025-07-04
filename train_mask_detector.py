import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

learning_rate = 1e-4
total_epochs = 20
batch_sz = 32

data_dir = r"C:\Users\Hafsa Fatima\Downloads\mask_detector\dataset"
categories = ["with_mask", "without_mask"]

images = []
targets = []

for category in categories:
    category_path = os.path.join(data_dir, category)
    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        image = load_img(image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        images.append(image_array)
        targets.append(category)

label_binarizer = LabelBinarizer()
targets = label_binarizer.fit_transform(targets)
targets = to_categorical(targets)

images = np.array(images, dtype="float32")
targets = np.array(targets)

(X_train, X_test, y_train, y_test) = train_test_split(
    images, targets, test_size=0.2, stratify=targets, random_state=42
)

augment = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

base_cnn = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_cnn.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(2, activation="softmax")(x)

mask_classifier = Model(inputs=base_cnn.input, outputs=x)

for layer in base_cnn.layers:
    layer.trainable = False

optimizer = Adam(learning_rate=learning_rate)
mask_classifier.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = mask_classifier.fit(
    augment.flow(X_train, y_train, batch_size=batch_sz),
    steps_per_epoch=len(X_train) // batch_sz,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // batch_sz,
    epochs=total_epochs
)

preds = mask_classifier.predict(X_test, batch_size=batch_sz)
predicted = np.argmax(preds, axis=1)

print(classification_report(y_test.argmax(axis=1), predicted, target_names=label_binarizer.classes_))

mask_classifier.save("mask_detector.keras")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, total_epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, total_epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, total_epochs), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, total_epochs), history.history["val_accuracy"], label="val_acc")
plt.title("Loss & Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
