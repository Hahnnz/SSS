import tensorflow as tf

def conv(data, ksize, filters, ssize, padding, use_bias, conv_name=None, bn_name=None, bn=False, act=True):
    if not bn :
        if act : 
            output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                      strides=(ssize,ssize),
                                      padding=padding.upper(),
                                      name=conv_name, 
                                      activation=tf.nn.relu,use_bias=use_bias,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        else : 
            output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                      strides=(ssize,ssize),
                                      padding=padding.upper(),
                                      name=conv_name,use_bias=use_bias,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    else : 
        conv = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                strides=(ssize,ssize),
                                padding=padding.upper(),
                                name=conv_name,use_bias=use_bias,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope(bn_name) as bn_name:
            output = tf.contrib.layers.batch_norm(conv)
        if act : output = tf.nn.relu(output)
    return output

def deconv(data, ksize, filters, ssize, padding, use_bias, deconv_name=None, bn_name=None, bn=False, act=True):
    if not bn :
        if act : 
            output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                                strides=(ssize,ssize),
                                                padding=padding,
                                                name=deconv_name, 
                                                activation=tf.nn.relu,use_bias=use_bias,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        else : 
            output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                                strides=(ssize,ssize),
                                                padding=padding,
                                                name=deconv_name,use_bias=use_bias,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer())
    else : 
        deconv = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                          strides=(ssize,ssize),
                                          padding=padding,
                                          name=deconv_name,use_bias=use_bias,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope(bn_name) as bn_name:
            output = tf.contrib.layers.batch_norm(deconv)
        if act : output = tf.nn.relu(output)
    return output

def max_pooling(data, ksize=3, ssize=2, name=None):
    return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)

def dropout(data, ratio, name=None):
    return tf.nn.dropout(data, ratio, name=name)


def bn(data, name=None):
    with tf.variable_scope(name) as name:
        batch_norm = tf.contrib.layers.batch_norm(data)
    return batch_norm
