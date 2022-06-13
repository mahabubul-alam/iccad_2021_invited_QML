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
outdir = f'AE_results/{out_prefix}'
os.makedirs(outdir, exist_ok = True)

#build qnn model
qubit = 8
layers = 4
enc = 4
pqc = 14
meas = 3
qnn = qnn_builder.PennylaneQNNCircuit(enc = enc, qubit = qubit, layers = layers, pqc = pqc, meas = meas)

#check compatibility between the chosen model and the dataset
input_length = qnn.enc_builder.max_inputs_length()
assert features <= input_length

#build the quantum-classical hybrid learning model
weight_shape = qnn.pqc_builder.weigths_shape()
if isinstance(weight_shape[0], tuple):
    ql_weights_shape = {'weights0': weight_shape[0], 'weights1': weight_shape[1]}
else:
    ql_weights_shape = {'weights0': weight_shape, 'weights1': ()}
output_dim = qnn.meas_builder.output_dim()

dev = qml.device("default.qubit", wires = qubit) #target pennylane device
qnode = qml.QNode(qnn.construct_qnn_circuit, dev, interface = 'tf', diff_method = 'backprop') #circuit

qlayer = qml.qnn.KerasLayer(qnode, ql_weights_shape, output_dim = output_dim)
clayer = tf.keras.layers.Dense(classes)
model = tf.keras.models.Sequential([qlayer, clayer])

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
