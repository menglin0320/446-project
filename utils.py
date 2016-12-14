import numpy as np
import scipy.misc
import os
from datetime import datetime as dt
import argparse
from models import VGG16, I2V
import tensorflow as tf

def getModel(image, params_path, model):
    if model == 'vgg':
        return VGG16(image, params_path)
    elif model == 'i2v':
        return I2V(image, params_path)
    else:
        print 'Invalid model name: use `vgg` or `i2v`'
        return None

class rnn_part(object):
    def __init__(self,rnn_in,seq_length,seq_per_image,num_rnn_units,lengths,labels,initial_state):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        with tf.variable_scope('Rnn'):
            shape = rnn_in.get_shape()
            map_size = int(shape[1])
            self.aug_batch = tf.transpose(tf.reshape(tfrepeat(rnn_in,seq_per_image),[seq_per_image,-1,map_size]),perm=[1,0,2])
            self.aug_batch = tf.reshape(self.aug_batch,[-1,1,map_size])
            print 'aug_batch:' + str(self.aug_batch.get_shape())

            augmented_batch_size = int(self.aug_batch.get_shape()[0])
            zero_holders = tf.zeros([augmented_batch_size,seq_length,map_size],dtype=tf.float32)
            print 'zero_holders:' + str(zero_holders.get_shape())

            seq_batches = tf.concat(1,[self.aug_batch,zero_holders])

            num_classes = int(labels.get_shape()[2])
            print 'labels' + str(labels.get_shape())
            aug_label = tf.concat(1,[tf.zeros([augmented_batch_size,1,num_classes],dtype=tf.float32),labels])
            self.x = tf.concat(2,[seq_batches,aug_label])
            print 'rnn x:' + str(self.x.get_shape())
            # Define a lstm cell with tensorflow
            lstm_cell = LayerNormalizedLSTMCell(num_rnn_units)
            self.outputs, self.state  = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                dtype=tf.float32,
                sequence_length=lengths,
                inputs=self.x,
                initial_state=initial_state)
            self.outputs = tf.reshape(self.outputs,[-1,lstm_cell.output_size])

def weight_variable(shape,method = 'xavier',name = '',std = 0.01):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    #initial = tf.random_normal(shape, mean=0.0, stddev=0.00)
    if method == "zeros":
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    elif method == "xavier":
        return tf.get_variable(name, shape=shape,
           initializer=tf.contrib.layers.xavier_initializer())
    else: 
        return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=std))

def nn_layer(input_tensor, input_dim, output_dim, name, act=tf.nn.relu,method = "xavier"):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(name):
        weights = weight_variable([input_dim, output_dim],method = method,name = name)
        bias = tf.Variable(tf.zeros(output_dim))    
        preactivate = tf.matmul(input_tensor, weights) + bias
        tf.histogram_summary(name + '/pre_activations', preactivate)
        if act is None:
          activations = preactivate
        else:
          activations = act(preactivate, 'activation')
    return activations

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    for i in range(0,n_labels):
        labels_one_hot[i,int(labels[i])] = 1
    return labels_one_hot

def tfrepeat(a, repeats):
  return tf.tile(a, [repeats,1])

def make_mask(lengths,array_shape):
    mask = np.full(array_shape, False, dtype=bool)
    for i in range(0,lengths.shape[0]):
        mask[0:int(lengths[i])] = True
    return mask

def shape_data2batchsize(data,batch_size,seq_per_image):
    zeros_length = batch_size - 1
    zeros_shape = np.array(data['images'].shape)
    zeros_shape[0] = zeros_length 
    data['images'] = np.concatenate((data['images'], np.zeros(zeros_shape, dtype=np.float32)), axis=0)
    #Labels
    label_zeros_shape = np.array( data['labels'].shape)
    label_zeros_shape[0] = zeros_length*seq_per_image
    data['labels'] = np.concatenate((data['labels'], np.zeros(label_zeros_shape, dtype=np.float32)), axis=0)
    print(data['labels'].shape)
    
    lengths_zeros_shape =  np.array(data['length'].shape)
    lengths_zeros_shape[0] = zeros_length*seq_per_image
    data['length'] = np.concatenate((data['length'], np.zeros(lengths_zeros_shape, dtype=np.float32)), axis=0)
    print(data['length'].shape)
    return data
def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = tf.nn.rnn_cell._linear([inputs, h], 4 * self._num_units, False)

            i, j, f, o = tf.split(1, 4, concat)

            # add layer normalization to each gate
            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state    