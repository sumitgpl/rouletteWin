import tkinter as tk
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from math import floor, ceil
#from pylab import rcParams
import model
import sys

chkfile ="checkpoint/winnumber.ckpt"

class MyModelGUI:
    def __init__(self, master):
        self.master = master
        master.title("Roulette wheel prediction")
        master.geometry('350x400')
        speedlist = ['slow','medium','fast']
        rotationlist = ['clock','anticlock']
        whnum = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]

        self.sess,self.predictions,self.x,self.keep_prob = self.getsession()

        self.label = tk.Label(master, text="Wheel Rotation",justify=tk.LEFT)
        self.label.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label.place(x = 10, y = 30, width=85, height=25)
        
        self.wrvar = tk.StringVar()
        self.wr = tk.OptionMenu(master,self.wrvar,*rotationlist)
        self.wr.place(x = 200, y = 30, width=150, height=25)  
        #self.wr = tk.Entry(master)
      
        self.label2 = tk.Label(master, text="Wheel Rotation Speed",justify=tk.LEFT)
        self.label2.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label2.place(x = 10, y = 60, width=120, height=25)  

        self.wsvar = tk.StringVar()
        self.ws = tk.OptionMenu(master,self.wsvar,*speedlist)
        self.ws.place(x = 200, y = 60, width=150, height=25)         

        self.label3 = tk.Label(master, text="Ball Rotation",justify=tk.LEFT)
        self.label3.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label3.place(x = 10, y = 90, width=75, height=25)      

        self.brvar = tk.StringVar()
        self.br = tk.OptionMenu(master,self.brvar,*rotationlist)
        self.br.place(x = 200, y = 90, width=150, height=25)  

        self.label4 = tk.Label(master, text="Ball Rotation Speed",justify=tk.LEFT)
        self.label4.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label4.place(x = 10, y = 120, width=110, height=25) 

        self.bsvar = tk.StringVar()
        self.bs = tk.OptionMenu(master,self.bsvar,*speedlist)
        self.bs.place(x = 200, y = 120, width=150, height=25) 

        self.label5 = tk.Label(master, text="Focus Area Number",justify=tk.LEFT)
        self.label5.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label5.place(x = 10, y = 150, width=115, height=25)    

        self.fcvar = tk.StringVar()
        self.ws = tk.OptionMenu(master,self.fcvar,*whnum)
        self.ws.place(x = 200, y = 150, width=150, height=25) 


        self.greet_button = tk.Button(master, text="Predict !", command=self.predict)
        self.greet_button.place(x = 120, y = 180, width=115, height=25)  

        self.label6 = tk.Label(master, text="Predict Number 1",justify=tk.LEFT)
        self.label6.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label6.place(x = 10, y = 230, width=100, height=25)   

        self.pdk1 = tk.Label(master, text="0",justify=tk.LEFT, borderwidth=2, relief="groove")
        self.pdk1.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.pdk1.place(x = 200, y = 230, width=150, height=25)          

        self.label7 = tk.Label(master, text="Predict Number 2",justify=tk.LEFT)
        self.label7.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label7.place(x = 10, y = 270, width=100, height=25)   

        self.pdk2 = tk.Label(master, text="0",justify=tk.LEFT, borderwidth=2, relief="groove")
        self.pdk2.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.pdk2.place(x = 200, y = 270, width=150, height=25)                  

        self.label8 = tk.Label(master, text="Actual Number",justify=tk.LEFT)
        self.label8.pack(side=tk.LEFT,  expand=tk.YES, fill=tk.BOTH)
        self.label8.place(x = 10, y = 300, width=90, height=25)      

        self.acvar = tk.StringVar()
        self.ws = tk.OptionMenu(master,self.acvar,*whnum)
        self.ws.place(x = 200, y = 300, width=150, height=25)          

        self.save_button = tk.Button(master, text="Save Data !", command=self.predict)
        self.save_button.place(x = 120, y = 330, width=115, height=25)  

    def predict(self):
        # print("self.wr->",self.wrvar.get())
        # print("self.ws->",self.wsvar.get())
        # print("self.br->",self.brvar.get())
        # print("self.bs->",self.bsvar.get())
        # print("self.fcvar->",self.fcvar.get())

        data = {'wheelrotation':[self.wrvar.get()],
                'wheelspeed':[self.wsvar.get()],
                'ballrotation':[self.brvar.get()],
                'ballspeed':[self.bsvar.get()],
                'startingpoint':[int(self.fcvar.get())]}
        x_data = pd.DataFrame(data)
        x_data = model.inpur_format(x_data)
        feed_dict ={self.x:x_data,self.keep_prob: 0.9}
        predict_out1 = self.sess.run(tf.argmax(self.predictions, 1), feed_dict=feed_dict)
        predict_out2 = self.sess.run(tf.argmin(self.predictions, 1), feed_dict=feed_dict)
        self.pdk1['text']=model.WHEELCIRCLE[predict_out1[0]]
        self.pdk1['bg'] = 'yellow'
        self.pdk2['text']=model.WHEELCIRCLE[predict_out2[0]]
        self.pdk2['bg'] = 'yellow'


    def getsession(self):
        ##creating network & weights #network load
        n_hidden_1 = 256
        n_input = 47
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
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, chkfile)
        print("Model restored.")
        return sess,predictions,x,keep_prob

root = tk.Tk()
my_gui = MyModelGUI(root)
root.mainloop()