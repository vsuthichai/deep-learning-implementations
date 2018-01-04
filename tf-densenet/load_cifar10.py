import pickle
import numpy as np
from six.moves import xrange

cifar10_train_data = None
cifar10_train_labels = None
cifar10_test_data = None
cifar10_test_labels = None

def load_cifar10():
    global cifar10_train_data, cifar10_train_labels, cifar10_test_data, cifar10_test_labels

    with open("cifar-10-batches-py/data_batch_1", "rb") as f:
        db1 = pickle.load(f, encoding='bytes')
    with open("cifar-10-batches-py/data_batch_2", "rb") as f:
        db2 = pickle.load(f, encoding='bytes')
    with open("cifar-10-batches-py/data_batch_3", "rb") as f:
        db3 = pickle.load(f, encoding='bytes')
    with open("cifar-10-batches-py/data_batch_4", "rb") as f:
        db4 = pickle.load(f, encoding='bytes')
    with open("cifar-10-batches-py/data_batch_5", "rb") as f:
        db5 = pickle.load(f, encoding='bytes')
        
    cifar10_train_data = np.concatenate([db1[b'data'], db2[b'data'], db3[b'data'], db4[b'data'], db5[b'data']], axis=0)
    cifar10_train_labels = np.array(db1[b'labels'] + db2[b'labels'] + db3[b'labels'] + db4[b'labels'] + db5[b'labels'])

    with open("cifar-10-batches-py/test_batch", "rb") as f:
        db1 = pickle.load(f, encoding='bytes')

    cifar10_test_data = np.array(db1[b'data'])
    cifar10_test_labels = np.array(db1[b'labels'])


def generate_train_batch(batch_size=64):
    global cifar10_train_data, cifar10_train_labels

    if cifar10_train_data is None or cifar10_train_labels is None:
        load_cifar10()

    indices = np.arange(len(cifar10_train_labels))
    np.random.shuffle(indices)
    data = cifar10_train_data[indices]
    labels = cifar10_train_labels[indices]

    for batch_index in xrange(0, len(cifar10_train_labels), batch_size):
        yield data[batch_index : batch_index + batch_size], labels[batch_index : batch_index + batch_size]


def generate_test_batch(batch_size):
    global cifar10_test_data, cifar10_test_labels

    if cifar10_test_data is None or cifar10_test_labels is None:
        load_cifar10()

    indices = np.arange(len(cifar10_test_labels))
    np.random.shuffle(indices)
    data = cifar10_test_data[indices]
    labels = cifar10_test_labels[indices]

    for batch_index in xrange(0, len(cifar10_test_labels), batch_size):
        yield data[batch_index : batch_index + batch_size], labels[batch_index : batch_index + batch_size]

__all__ = ['generate_train_batch', 'generate_test_batch']

