import pandas as pd
from tensorflow.python.keras.utils import np_utils

def readucr(filename):
    data = pd.read_csv(filename, header=None, encoding='utf-8')
    Y = data[[0]].values
    data.drop(data.columns[0], axis=1, inplace=True)
    X = data.values
    return X, Y

noiseroot = "/path/to/your/dataset/LQ/"
root = "/path/to/your/dataset/HQ/"
nb_classes = 63
x_train_t, y_train = readucr(root+'t_train.csv')
x_val_t, y_val = readucr(root+'t_test.csv')
x_test_t, y_test = readucr(noiseroot+'tn_test.csv')

x_train_k, y_train_k = readucr(root+'k_train.csv')
x_val_k, y_val_k = readucr(root+'k_test.csv')
x_test_k, y_test_k = readucr(noiseroot+'kn_test.csv')

x_train_v, y_train_v = readucr(root+'v_train.csv')
x_val_v, y_val_v = readucr(root+'v_test.csv')
x_test_v, y_test_v = readucr(noiseroot+'vn_test.csv')


print(y_train)
# Verify label consistency across modalities
if (y_train == y_train_k).all():
    if (y_train == y_train_v).all():
        print('train tag ok')
    else:
        raise Exception('y_train_h != y_train_v')
else:
    raise Exception('y_train_h != y_train_k')

if (y_val == y_val_k).all():
    if(y_val == y_val_v).all():
        print('val tag ok')
    else:
        raise Exception('y_val_h != y_val_v')
else:
    raise Exception('y_val_h != y_val_k')

if (y_test == y_test_k).all():
    if(y_test == y_test_v).all():
        print('test tag ok')
    else:
        raise Exception('y_test_h != y_test_v')
else:
    raise Exception('y_test_h != y_test_k')


def Y_process(Y_data):
    y = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min()) * (nb_classes - 1)
    y = np_utils.to_categorical(y, nb_classes)
    return y
Y_train = Y_process(y_train)
Y_val = Y_process(y_val)
Y_test = Y_process(y_test)

def X_process(X_data):
    x_mean = X_data.mean()
    x_std = X_data.std()
    x = (X_data - x_mean) / (x_std)
    x = X_data.reshape(X_data.shape + (1, ))
    return x

# Standardise tactile, kinesthetic data
X1_train = X_process(x_train_t)
X1_val = X_process(x_val_t)
X1_test = X_process(x_test_t)
X2_train = X_process(x_train_k)
X2_val = X_process(x_val_k)
X2_test = X_process(x_test_k)
# visual
X3_train = x_train_v.reshape(x_train_v.shape + (1, ))
X3_val = x_val_v.reshape(x_val_v.shape + (1, ))
X3_test = x_test_v.reshape(x_test_v.shape + (1, ))
print(len(X1_train))
print(len(X2_train))
print(len(X3_train))



