import tensorflow as tf

def one(x):
    return tf.constant(1.0)

def zero(x):
    return tf.constant(0.0)

def add(x, y):
    return tf.divide(tf.add(x, y), 2.0)

def sqrt(x):
    return tf.sqrt(tf.abs(x))

def ypow(x):
    return tf.pow(tf.abs(x), tf.abs(y))

def expx(x):
    return tf.divide(tf.subtract(tf.exp(x), 1.0), tf.subtract(tf.exp(1.0), 1.0))

def sqrtxy(x, y):
    return tf.divide(tf.sqrt(tf.add(tf.multiply(x, x), tf.multiply(y, y))), tf.sqrt(2.0))

def aminus(x, y):
    return tf.divide(tf.abs(tf.subtract(x, y)), 2.0)

def sigmoid(x):
    return tf.divide(1.0, tf.add(1.0, tf.exp(tf.multiply(-1.0, x))))

## MTCGP functions

def reduce_size(x, y):
    while len(x.shape) > len(y.shape):
        x = tf.reduce_mean(x, axis=1)
    while len(y.shape) > len(x.shape):
        y = tf.reduce_mean(x, axis=1)
    return x, y

def first(x):
    if len(x.shape) > 1:
        return x[:, 0]
    return x

def last(x):
    if len(x.shape) > 1:
        return x[:, -1]
    return x

def reduce_max(x):
    if len(x.shape) > 1:
        return tf.reduce_max(x, axis=1)
    return x

def reduce_min(x):
    if len(x.shape) > 1:
        return tf.reduce_min(x, axis=1)
    return x

def reduce_mean(x):
    if len(x.shape) > 1:
        return tf.reduce_mean(x, axis=1)
    return x

def reduce_sum(x):
    if len(x.shape) > 1:
        return tf.reduce_sum(x, axis=1)
    return x

def reduce_prod(x):
    if len(x.shape) > 1:
        return tf.reduce_prod(x, axis=1)
    return x
