from lib.model.cnnBiGRUbisa import build_T, WeightedFusionUnit
from lib.data.Dataloader import * # X1_train, X2_train, X3_train, X1_val, X2_val, X3_val, X1_test, X2_test, X3_test, Y_train, Y_val, Y_test, nb_classes
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import time
import os
import random
import tensorflow as tf
import numpy as np
from sklearn import metrics
import pandas as pd

from clr_callback import CyclicLR


CUDA_VISIBLE_DEVICES = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
physical_device = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)

lab_name = 'RobustVisH-AU'



nb_epochs = 100
batch_size = 8
model1 = build_T(X1_train, X2_train, X3_train, nb_classes)  # SA model
model1.summary()
print(lab_name)
print("cuda: ", CUDA_VISIBLE_DEVICES)

clr_triangular = CyclicLR(base_lr=0.0001, max_lr=0.01, mode='triangular2')
optimizer = keras.optimizers.Adam()
model1.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
callbacks = [
    ModelCheckpoint(lab_name+'.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_freq='epoch'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001),
    clr_triangular,
    TensorBoard(log_dir=lab_name + '_tensorboard_logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    CSVLogger(lab_name+'.log', separator=",", append=False)
]
print('tactile shape:', X1_train.shape, '\n kinesthetics shape:', X2_train.shape, '\n visual shape:', X3_train.shape,  '\n y shape:', Y_train.shape)
start_time=time.time()
history = model1.fit([X1_train, X2_train, X3_train], Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=([X1_val, X2_val, X3_val], Y_val), callbacks=callbacks)
end_time=time.time()
train_time_cost=end_time-start_time

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

result=pd.DataFrame(history.history)
result.to_csv(lab_name+".csv", index=False)


epochs = range(len(acc))

start_time=time.time()
score = model1.evaluate([X1_test, X2_test, X3_test], Y_test)
end_time=time.time()
test_time_cost=end_time-start_time
print("Accuracy after loading Model:", score[1]*100)
list = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), score[0], score[1]]


# Show confusion matrix
def F1recall(model, x_val, y_val):
    predictions = model.predict(x_val)
    predictions = predictions.argmax(axis=1)
    truelabel = y_val.argmax(axis=1)  # one-hot to label
    recall = metrics.recall_score(truelabel, predictions, average='weighted')
    f1_score = metrics.f1_score(truelabel, predictions, average='weighted')
    print("recall:", recall)
    print("f1_score:", f1_score)
    list.append(f1_score)
    list.append(recall)


F1recall(model1, [X1_test, X2_test, X3_test], Y_test)
list.append(train_time_cost)
list.append(test_time_cost)
df = pd.DataFrame(columns=['test_date', 'loss', 'accuracy','f1_score','recall','time_cost(train)','time_cost(val)'])  # 列名
df.to_csv(lab_name+"_test(runtrain).csv", index=False)
print(list)
data = pd.DataFrame([list])
data.to_csv(lab_name+"_test(runtrain).csv", mode='a', header=False, index=False)




model_test = keras.models.load_model(lab_name+'.h5', custom_objects={'WeightedFusionUnit': WeightedFusionUnit})
predictions = model_test.predict([X1_test, X2_test, X3_test])
pred_labels = predictions.argmax(axis=1)
true_labels = Y_test.argmax(axis=1)
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)

print(f"{lab_name.split('/')[-1]} Accuracy: {accuracy:.4f}")
print(f"{lab_name.split('/')[-1]} Recall: {recall:.4f}")
print(f"{lab_name.split('/')[-1]} F1 Score: {f1:.4f}")
print(f"{lab_name.split('/')[-1]} Precision: {precision:.4f}")