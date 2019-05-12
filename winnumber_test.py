import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from math import floor, ceil
#from pylab import rcParams
import model
import sys

#%matplotlib inline

chkfile ="checkpoint/winnumber.ckpt"
#metafile = "checkpoint/winnumber.ckpt.meta"

def encode(series): 
  return pd.get_dummies(series.astype(str))

#test data 
#clock,medium,anticlock,medium,29 ,15
data = {'wheelrotation':['clock'] ,
        'wheelspeed':['medium'],
        'ballrotation':['anticlock'],
        'ballspeed':['medium'],
        'startingpoint':[29]}
x_data = pd.DataFrame(data)
print(x_data)

#x_data.columns=['wheelrotation','wheelspeed','ballrotation','ballspeed','startingpoint']
x_data = model.inpur_format(x_data)
print(x_data.shape)
sys.exit
##creating network & weights #network load
n_hidden_1 = 256
n_input = x_data.shape[1]
n_classes = 37

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

training_epochs = 1000
display_step = 100
batch_size = 32
x = tf.placeholder("float", [None, n_input],name="x")
y = tf.placeholder("float", [None, n_classes] ,name="y")

predictions = model.multilayer_perceptron(x, weights, biases, keep_prob)
#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

#tf.reset_default_graph()  
#saver = tf.train.import_meta_graph(chkfile+".meta")
#print("saver-->",saver)



#var_name_list---> ['h1:0', 'h2:0', 'h3:0', 'out:0', 'b1:0', 'b2:0', 'b3:0', 'out_1:0']

#tf.reset_default_graph()
# Create some variables.
#x_data =[[9, 0, 1 ,1 ,0 ,0 ,0 ,1,0 ,1 ,0]] #[22]
#v1 = tf.get_variable('Variable:0', shape=[11])
#w2 = graph.get_tensor_by_name("w2:0")

#x = tf.get_variable("x", shape=[11])
#keep_prob = tf.placeholder("float")
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.

with tf.Session() as sess:

  # Restore variables from disk.
  #saver.restore(sess,tf.train.latest_checkpoint(chkfile))
    saver.restore(sess, chkfile)
    print("Model restored.")
    #graph = tf.get_default_graph()
    #x = graph.get_tensor_by_name("x:0")
    #y = graph.get_tensor_by_name("y:0")
    #bias = graph.get_tensor_by_name("out_1:0")
    #print('x-->',x)
    #print('y-->',y)
   
    #loss=graph.get_tensor_by_name("Mean:0")
    #w2 = graph.get_tensor_by_name("h2:0")
    #w2 = graph.get_tensor_by_name("h3:0")
    feed_dict ={x:x_data,keep_prob: 0.9}  
    #tf.global_variables_initializer().run(session =sess)
  #var_name_list = [v.name for v in tf.trainable_variables()]
  #print('var_name_list-->',var_name_list)
    #add_on_op =  y
    predict_out = sess.run(tf.argmax(predictions, 1), feed_dict=feed_dict)
    predict_out2 = sess.run(tf.argmin(predictions, 1), feed_dict=feed_dict)
    print('predict_out-->',predict_out)
    print('predict_out2-->',predict_out2)
    #predict_out = sess.run([predictions], feed_dict=feed_dict)
    #int(sess.run(predictions))
    print("predictions--->",model.WHEELCIRCLE[predict_out[0]])
    print("predictions 2--->",model.WHEELCIRCLE[predict_out2[0]])
    #print(tf.argmax(sess.run(y,feed_dict),1))
    #add_on_op = y*bias
    #print('output-->',add_on_op)
    #print(sess.run(tf.all_variables()))
  #print("v1 : %s" % v1.eval())
  # Check the values of the variables
  #print("v1 : %s" % v1.eval())
  #print("v2 : %s" % v2.eval())