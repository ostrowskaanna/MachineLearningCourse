import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load("tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True,
with_info=True)

info

class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224]) 
    return resized_image, label

import tensorflow as tf
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./127.5, offset=-1))

filters = [32, 64, 128, 256]
sizes = [7, 6, 5, 4]

for f, s in zip(filters, sizes):
    model.add(tf.keras.layers.Conv2D(filters=f, kernel_size=s, strides=1, activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128 , activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=64 , activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=5, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_set, validation_data=valid_set, epochs=10)

acc_train = model.evaluate(train_set)
acc_test = model.evaluate(test_set)
acc_valid = model.evaluate(valid_set)
acc = (acc_train, acc_valid, acc_test)
print(acc)

import pickle
with open("simple_cnn_acc.pkl", "wb") as file:
    pickle.dump(acc, file)

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1) 
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(train_set, validation_data=valid_set, epochs=5)

for layer in base_model.layers:
    layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(train_set, validation_data=valid_set,epochs=10)

acc_train = model.evaluate(train_set)
acc_test = model.evaluate(test_set)
acc_valid = model.evaluate(valid_set)
acc = (acc_train, acc_valid, acc_test)
print(acc)

with open("xception_acc.pkl", "wb") as file:
    pickle.dump(acc, file)


