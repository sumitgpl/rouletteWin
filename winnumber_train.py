from __future__ import print_function

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
#import seaborn as sns
from math import floor, ceil
#from pylab import rcParams
import model
import sys
from sklearn.utils import shuffle

#%matplotlib inline

chkfile ="checkpoint/winnumber.ckpt"

def encode(series): 
  return pd.get_dummies(series.astype(str))

# def multilayer_perceptron(x, weights, biases, keep_prob):
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)

#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
    
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     layer_3 = tf.nn.relu(layer_3)
#     layer_3 = tf.nn.dropout(layer_3, keep_prob)
#     out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#     return out_layer

## data loading and preparing 
data_csv = pd.read_csv("roulettedata.csv", sep=",")
#data_csv = shuffle(data_csv)
#print(data_csv)
#print(data_csv.shape)

#data_x['winnumber'] =data_csv.winnumber
data_x = model.inpur_format(data_csv)
print(data_x)
print(data_x.shape)

#data_y = pd.DataFrame()
#data_y['winnumber'] =data_csv.winnumber
data_y = model.out_format(data_csv)

print(data_y)
print(data_y.shape)

##Splitting the data
train_size = 0.8
train_cnt = floor(data_x.shape[0] * train_size)
x_train = data_x.iloc[0:train_cnt].values
y_train = data_y.iloc[0:train_cnt].values
x_test = data_x.iloc[train_cnt:].values
y_test = data_y.iloc[train_cnt:].values

print("x_test-->",x_test[0])
print("x_test-->",y_test[0])

##creating network & weights
# Parameters

n_hidden_1 = 256
n_input = data_x.shape[1]
n_classes = data_y.shape[1]
#0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
cat_features = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="h1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1]),name="h2"),
    'h3': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1]),name="h3"),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]),name="out")
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]),name="b1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_1]),name="b2"),
    'b3': tf.Variable(tf.random_normal([n_hidden_1]),name="b3"),
    'out': tf.Variable(tf.random_normal([n_classes]),name="out")
}
keep_prob = tf.placeholder("float",name="keep_prob")

training_epochs = 5000
display_step = 100
batch_size = 10
x = tf.placeholder("float", [None, n_input],name="x")
y = tf.placeholder("float", [None, n_classes] ,name="y")

#network load
predictions = model.multilayer_perceptron(x, weights, biases, keep_prob)

#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

##running training 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.0
        epoch_loss = 0
        total_batch = int(len(x_train) / batch_size)
        #print('len(x_train)-->',len(x_train))
        #print('batch_size-->',batch_size)
        #print('total_batch-->',total_batch)

        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # _, c = sess.run([optimizer, cost], 
            #                 feed_dict={
            #                     x: batch_x,
            #                     y: batch_y, 
            #                     keep_prob: 0.8
            #                     })
            # avg_cost += c / total_batch
            # epoch_loss += c
        if epoch % display_step == 0 or epoch == 1: 
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})
            print("Step " + str(epoch) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + 
            "{:.3f}".format(acc))

            #print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            #print('Epoch', epoch, 'loss:',epoch_loss)
    print("Optimization Finished!")
    #print("Predicted value-->",tf.argmax(predictions, 1))
    #print("Actual value-->",tf.argmax(y, 1))
    #correct_prediction = tf.equal(tf.argmax(predictions, 1),tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # Calculate accuracy
    print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: x_test,y: y_test}))

    print("x:",x.shape)
    print("y:",y.shape)
    print("x_test:",x_test.shape)
    print("y_test:",y_test.shape)
    #print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))
    save_path = saver.save(sess, chkfile)
    print("Model saved in path: %s" % save_path)
    var_name_list = [v.name for v in tf.trainable_variables()]
    print('var_name_list--->',var_name_list)