from sklearn.datasets import load_breast_cancer
import app.qnn_builder as qnn_builder
import pennylane as qml
import tensorflow as tf
from sklearn.preprocessing import normalize
from numpy import genfromtxt
import sys
import os
import numpy as np

data = genfromtxt(sys.argv[1], delimiter=',')
data = data[:2000]
X = data[:,:-1]
X = normalize(X, axis = 0, norm = 'max')
y = data[:,-1]
features = X.shape[-1]
classes = max(y) + 1

out_prefix = sys.argv[1].split('/')[-1].split('.')[0]
outdir = f'AE_Classical_results/{out_prefix}'
os.makedirs(outdir, exist_ok = True)

c1layer = tf.keras.layers.Dense(features)
actlayer = tf.keras.layers.Activation('LeakyReLU')
c2layer = tf.keras.layers.Dense(classes)
model = tf.keras.models.Sequential([c1layer, actlayer, c2layer])

#compile and train the model
opt = tf.keras.optimizers.SGD(learning_rate = 0.5)
model.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
history_callback = model.fit(X, y, epochs = 20, batch_size = 50, validation_split = 0.5, use_multiprocessing = True)

tr_loss_history = history_callback.history['loss']
tr_acc_history = history_callback.history['accuracy']
val_loss_history = history_callback.history['val_loss']
val_acc_history = history_callback.history['val_accuracy']
np.savetxt(
    f'{outdir}/tr_loss_history.csv', np.array(tr_loss_history), delimiter=","
)

np.savetxt(
    f'{outdir}/tr_acc_history.csv', np.array(tr_acc_history), delimiter=","
)

np.savetxt(
    f'{outdir}/val_loss_history.csv', np.array(val_loss_history), delimiter=","
)

np.savetxt(
    f'{outdir}/val_acc_history.csv', np.array(val_acc_history), delimiter=","
)
