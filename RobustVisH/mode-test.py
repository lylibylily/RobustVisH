import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
from keras_flops import get_flops
from lib.data.Dataloader import *
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 用于绘制混淆矩阵
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, classification_report
from tensorflow import keras
import numpy as np
from lib.model.cnnBiGRUbisa import WeightedFusionUnit
from sklearn import metrics

from sklearn.metrics import confusion_matrix, recall_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


lab_name='/path/to/your/model/folder/'+'RobustVisH-AU'
model_test = keras.models.load_model(lab_name+'.h5', custom_objects={'WeightedFusionUnit': WeightedFusionUnit})
model_test.summary()
total_params = np.sum([np.prod(w.shape) for w in model_test.weights])
flops = get_flops(model_test, batch_size=1)
print(f"{lab_name.split('/')[-1]} Params (M): {total_params / 10 ** 6}")
print(f"{lab_name.split('/')[-1]} GFLOPs: {flops / 10 ** 9}")

predictions = model_test.predict([X1_val, X2_val, X3_val])
pred_labels = predictions.argmax(axis=1)
true_labels = Y_val.argmax(axis=1)
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
print(f"{lab_name.split('/')[-1]} HQ Accuracy: {accuracy:.4f}")
print(f"{lab_name.split('/')[-1]} HQ Recall: {recall:.4f}")
print(f"{lab_name.split('/')[-1]} HQ F1 Score: {f1:.4f}")
print(f"{lab_name.split('/')[-1]} HQ Precision: {precision:.4f}")

predictions = model_test.predict([X1_test, X2_test, X3_test])
pred_labels = predictions.argmax(axis=1)
true_labels = Y_test.argmax(axis=1)
accuracy = accuracy_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
print(f"{lab_name.split('/')[-1]} LQ Accuracy: {accuracy:.4f}")
print(f"{lab_name.split('/')[-1]} LQ Recall: {recall:.4f}")
print(f"{lab_name.split('/')[-1]} LQ F1 Score: {f1:.4f}")
print(f"{lab_name.split('/')[-1]} LQ Precision: {precision:.4f}")

report = classification_report(true_labels, pred_labels, zero_division=0)