import tensorflow as tf
import pandas as pd

WHEELCIRCLE = [0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26]
WHEELSEQ = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
ROTATION = ['clock','anticlock']
SPEED = ['slow','medium','fast']

def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)
    #layer_3 = tf.nn.dropout(layer_3, keep_prob)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

def number_encrypt(y_series,collist =None):
    df_wc = pd.DataFrame(WHEELCIRCLE,columns=['wc'])
    df_ws = pd.DataFrame(WHEELSEQ,columns=['ws'])
    df =  pd.concat([df_ws,df_wc], axis=1)
    #df_y1= pd.get_dummies(df['ws'])
    df_2 = pd.concat([df,pd.get_dummies(df['ws'])], axis=1)
    y = y_series.to_frame(name='winnumber')
    df_new = y.merge(df_2,left_on='winnumber',right_on='wc', how='left', sort=False)
    #df_new = df_2.merge(y,left_on='wc',right_on='winnumber', how='left', sort=False)
    df_new= df_new.drop(columns=['winnumber','ws','wc'])
    if collist:
        df_new.columns  = collist
    return df_new

def rotation_encrypt(x_series,collist):
    df = pd.DataFrame(ROTATION,columns=['r'])
    df_2 = pd.concat([df,pd.get_dummies(df['r'])], axis=1)
    x = x_series.to_frame(name='wr')
    df_new = x.merge(df_2,left_on='wr',right_on='r', how='left')
    df_new=  df_new.drop(columns=['r','wr'] )
    df_new.columns  = collist
    return df_new

def speed_encrypt(x_series,collist):
    df = pd.DataFrame(SPEED,columns=['s'])
    df_2 = pd.concat([df,pd.get_dummies(df['s'])], axis=1)
    x = x_series.to_frame(name='ws')
    df_new = x.merge(df_2,left_on='ws',right_on='s', how='left')
    df_new= df_new.drop(columns=['s','ws'] , axis='columns')
    df_new.columns  = collist
    return df_new

def inpur_format(x):
    x_cols= ['X_0','X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10','X_11','X_12','X_13','X_14','X_15','X_16','X_17','X_18','X_19','X_20','X_21','X_22','X_23','X_24','X_25','X_26','X_27','X_28','X_29','X_30','X_31','X_32','X_33','X_34','X_35','X_36']
    data_x = pd.concat([
                number_encrypt(x.startingpoint,x_cols),
                rotation_encrypt(x.wheelrotation,['whanticlock','whclock']),
                speed_encrypt(x.wheelspeed,['whfast','whmedium','whslow']),
                rotation_encrypt(x.ballrotation,['banticlock','bclock']),
                speed_encrypt(x.ballspeed,['bfast','bmedium','bslow'])
                ], axis=1)
    return data_x
            
def out_format(y):
    return number_encrypt(y.winnumber)