import tensorflow as tf
import numpy as np
import cv2
import os
import h5py
import random

dataset_path = '../data/train.h5' 
valset_path = '../data/test.h5' 
model_path='../model/model.ckpt'


w=288
h=352
c=3

epoch=100
batch_size=16
lr = 0.00025
lr_decay = 1.0*lr/(epoch*800/batch_size)

def CNNnet(input_tensor):
    # INPUT 288*352*3
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[3,3,3,8],initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        conv1_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME") #144*176*16

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[3,3,8,8],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [8], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 72*88*16

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,8,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 36*44*16

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 18*22*64
        nodes = w*h/4/4/4/4*16
        reshaped = tf.reshape(pool4,[-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc1_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)


    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [16, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc2_biases = tf.get_variable("bias", [16], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)


    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [16, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        net = tf.matmul(fc2, fc3_weights) + fc3_biases
        
    prediction = tf.nn.softmax(net)

    return prediction


x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
y = CNNnet(x)

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def minibatch(data, label, batch_size):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        chunk = list(arr[start_idx:start_idx + batch_size])
        chunk.sort()
        yield data[chunk], label[chunk]

dataset = h5py.File(dataset_path, 'r')
data= dataset['data']
label = dataset['label']

valset = h5py.File(valset_path, 'r')
valdata= valset['data']
vallabel = valset['label']

saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
for epoch in range(epoch):
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatch(data, label, batch_size):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("epoch %d | train loss: %f | train acc: %f" % (epoch, np.sum(train_loss)/ n_batch, np.sum(train_acc)/ n_batch))

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatch(valdata, vallabel, batch_size):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("epoch %d | validation loss: %f | validation acc: %f" % (epoch, np.sum(val_loss)/ n_batch, np.sum(val_acc)/ n_batch))
    lr = lr -lr_decay
    saver.save(sess,model_path)
sess.close()

dataset.close()
valset.close()