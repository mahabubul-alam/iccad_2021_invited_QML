import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

torch.manual_seed(0)

n_qubits = 4
n_layers = 2
n_class = 3
n_features = 196
image_x_y_dim = 14
var_per_qubit = n_qubits
kernel_size = n_qubits 
stride = n_qubits

qnn_qubits = 10
qnn_layers = 4
assert qnn_qubits >= n_class

dev = qml.device("default.qubit", wires = n_qubits)
dev_qnn = qml.device("default.qubit", wires = qnn_qubits)

def circuit(inputs, weights):
    encoding_gates = ['RZ', 'RX']*int(var_per_qubit/2)
    for qub in range(n_qubits):
        qml.Hadamard(wires = qub)
        for i in range(var_per_qubit):
            if (qub * var_per_qubit + i) < len(inputs):
                exec(
                    f'qml.{encoding_gates[i]}({inputs[qub * var_per_qubit + i]}, wires = {qub})'
                )

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires = [i, (i + 1) % n_qubits])
            #qml.CNOT(wires = [i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires = j % n_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    #return qml.expval(qml.PauliZ(0))

def qnn_circuit(inputs, weights):
    encoding_gates = ['RZ', 'RX'] * (len(inputs) // 2)
    var_per_qubit = int(len(inputs)/qnn_qubits) + 1
    for qub in range(qnn_qubits):
        qml.Hadamard(wires = qub)
        for i in range(var_per_qubit):
            if (qub * var_per_qubit + i) < len(inputs):
                exec(
                    f'qml.{encoding_gates[i]}({np.pi * inputs[qub * var_per_qubit + i]}, wires = {qub})'
                )

    for l in range(qnn_layers):
        for i in range(qnn_qubits):
            qml.CRZ(weights[l, i], wires = [i, (i + 1) % qnn_qubits])
            #qml.CNOT(wires = [i, (i +n 1) % n_qubits])
        for j in range(qnn_qubits, 2 * qnn_qubits):
            qml.RY(weights[l, j], wires = j % qnn_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(qnn_qubits)]
    #return qml.expval(qml.PauliZ(0))


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 4, stride = 2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(16, 8, 4, stride = 2)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fl1 = nn.Flatten()
        self.ln1 = nn.LayerNorm(32, elementwise_affine = False)
        
        qnn_weight_shapes = {"weights": (qnn_layers, 2 * qnn_qubits)}
        qnn_qnode = qml.QNode(qnn_circuit, dev_qnn, interface = 'torch', diff_method = 'adjoint')
        self.ql1 = qml.qnn.TorchLayer(qnn_qnode, qnn_weight_shapes)
        
        self.fc1 = nn.Linear(qnn_qubits, n_class)

    def forward(self, X):
        bs = X.shape[0]
        X = X.view(bs, 1, image_x_y_dim, image_x_y_dim)
        X = self.conv1(X)
        X = self.lr1(X)
        X = self.conv2(X)
        X = self.lr2(X)
        X = self.fl1(X)
        X = self.ln1(X)

        X = self.ql1(X)
        X = self.fc1(X)
        return X

if __name__ == '__main__':
    network = Net()
    random_input = torch.rand(1, n_features)
    print(network(random_input))
