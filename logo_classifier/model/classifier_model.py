import tensorflow as tf

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, name='conv'):
    with tf.name_scope(name):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        act = tf.nn.relu(x)
        # tf.summary.histogram("weights", W)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
        return act


def maxpool2d(x, k=2, name='pool'):
    with tf.name_scope(name):
        # MaxPool2D wrapper
        pool= tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')
    return pool

def fc_layer(x, W, b, name='fc', act=tf.nn.relu):
    with tf.name_scope(name):
        fc1 = tf.add(tf.matmul(x, W), b)
        act = tf.nn.relu(fc1)
        # tf.summary.histogram("weights", W)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)
    return act

# Create model
def conv_net(x, weights, biases, dropout, img_w, img_h, img_d):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_w, img_h, img_d])

    # Convolution Layer
    conv1 = conv2d(x,     weights['wc1'], biases['bc1'], name="conv1")
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name="conv2")

    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2, name="pool1")

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], name="conv3")
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], name="conv4")

    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2, name="pool2")

    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], name="conv5")
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2, name="pool3")

    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'], name="conv6")


    # Max Pooling (down-sampling)
    conv6 = maxpool2d(conv6, k=2, name="pool4")
    conv6 = maxpool2d(conv6, k=2, name="pool5")

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    flatten = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = fc_layer(flatten, weights['wd1'], biases['bd1'], name="fc1")

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope("out"):
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out