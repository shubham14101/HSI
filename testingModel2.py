import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import *
import math
import re
import random
from tempfile import TemporaryFile



input_patch = tf.placeholder(tf.float32, [None, 5,5, 200])
ground_truth = tf.placeholder(tf.float32, [None,9])
'''
incp1 = tf.Variable(tf.truncated_normal([5,5,200, 128], dtype=tf.float32,stddev=1e-1),trainable=True)
incp2 = tf.Variable(tf.truncated_normal([3,3,200, 128], dtype=tf.float32,stddev=1e-1),trainable=True)
incp3 = tf.Variable(tf.truncated_normal([1,1,200, 128], dtype=tf.float32,stddev=1e-1),trainable=True)

convI1_5 = tf.nn.conv2d(input_patch,incp1, [1, 1, 1, 1], padding='SAME')
convI1_3 = tf.nn.conv2d(input_patch,incp2, [1, 1, 1, 1], padding='SAME')
convI1_1 = tf.nn.conv2d(input_patch,incp3, [1, 1, 1, 1], padding='SAME')

biasI1_5 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
outI1_5 = tf.nn.bias_add(convI1_5, biasI1_5)

biasI1_3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
outI1_3 = tf.nn.bias_add(convI1_3, biasI1_3)

biasI1_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
outI1_1 = tf.nn.bias_add(convI1_1, biasI1_1)

outI1_1 = tf.nn.max_pool(outI1_1,[1,5,5,1],[1,1,1,1],padding='SAME')
outI1_3 = tf.nn.max_pool(outI1_3,[1,3,3,1],[1,1,1,1],padding='SAME')


convI1=tf.concat([outI1_1,outI1_3,outI1_5],3)
convI1 = tf.nn.relu(convI1)
'''
Filter1 = tf.Variable(tf.truncated_normal([3,3,200, 128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(input_patch,Filter1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias1)
conv1_1 = tf.nn.relu(out)


Filter2 = tf.Variable(tf.truncated_normal([3,3,128, 128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(conv1_1,Filter2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias2)
conv1_2 = tf.nn.relu(out)

Filter3 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(conv1_2,Filter3, [1, 1, 1, 1], padding='SAME')
bias3 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias3)
conv1_3 = out

res1 = conv1_3 + conv1_1
res1 = tf.nn.relu(res1)


Filter4 = tf.Variable(tf.truncated_normal([3,3,128, 128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(res1,Filter4, [1, 1, 1, 1], padding='SAME')
bias4 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias4)
conv2_1 = tf.nn.relu(out)

Filter5 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(conv2_1,Filter5, [1, 1, 1, 1], padding='SAME')
bias5 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias5)
conv2_2 = out


res2 = conv1_1 + conv1_3 + conv2_2
res2 = tf.nn.relu(res2)

Filter6 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(res2,Filter6, [1, 1, 1, 1], padding='SAME')
bias6 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias6)
conv3_1 = tf.nn.relu(out)

Filter7 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(conv3_1,Filter7, [1, 1, 1, 1], padding='SAME')
bias7 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias7)
conv3_2 = out

res3 =  conv1_1 + conv1_3 + conv2_2 + conv3_2
res3 = tf.nn.relu(res3)

Filter8 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(res3,Filter8, [1, 1, 1, 1], padding='SAME')
bias8 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias8)
conv4_1 = tf.nn.relu(out)

Filter9 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
conv = tf.nn.conv2d(conv4_1,Filter9, [1, 1, 1, 1], padding='SAME')
bias9 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
out = tf.nn.bias_add(conv, bias9)
conv4_2 = tf.nn.relu(out)
##### side outputs
#1
sideFilter1_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias1_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv1_1out = tf.nn.conv2d(conv1_2,sideFilter1_1, [1, 1, 1, 1], padding='SAME')
conv1_1out = tf.nn.bias_add(conv1_1out,sideBias1_1)
conv1_1out = tf.nn.relu(conv1_1out)

sideFilter1_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias1_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv1_2out = tf.nn.conv2d(conv1_1out,sideFilter1_2, [1, 1, 1, 1], padding='SAME')
conv1_2out = tf.nn.bias_add(conv1_2out,sideBias1_2)
conv1_2out = tf.nn.relu(conv1_2out)

shape = int(np.prod(conv1_2out.get_shape()[1:]))
fcS1w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS1b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv1_2out, [-1, shape])
resultS1 = tf.nn.bias_add(tf.matmul(flat, fcS1w), fcS1b)
ansS1 = tf.nn.softmax(resultS1)
#2

sideFilter2_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias2_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv2_1out = tf.nn.conv2d(tf.nn.relu(conv1_3),sideFilter2_1, [1, 1, 1, 1], padding='SAME')
conv2_1out = tf.nn.bias_add(conv2_1out,sideBias2_1)

resS2 = conv1_1out + conv2_1out
resS2 = tf.nn.relu(resS2)

sideFilter2_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias2_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv2_2out = tf.nn.conv2d(resS2,sideFilter2_2, [1, 1, 1, 1], padding='SAME')
conv2_2out = tf.nn.bias_add(conv2_2out,sideBias2_2)
conv2_2out = tf.nn.relu(conv2_2out)

shape = int(np.prod(conv2_2out.get_shape()[1:]))
fcS2w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS2b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv2_2out, [-1, shape])
resultS2 = tf.nn.bias_add(tf.matmul(flat, fcS2w), fcS2b)
ansS2 = tf.nn.softmax(resultS2)
#3

sideFilter3_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias3_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv3_1out = tf.nn.conv2d(conv2_1,sideFilter3_1, [1, 1, 1, 1], padding='SAME')
conv3_1out = tf.nn.bias_add(conv3_1out,sideBias3_1)

resS3 = conv1_1out + conv2_1out + conv3_1out
resS3 = tf.nn.relu(resS3)

sideFilter3_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias3_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv3_2out = tf.nn.conv2d(resS3,sideFilter3_2, [1, 1, 1, 1], padding='SAME')
conv3_2out = tf.nn.bias_add(conv3_2out,sideBias3_2)
conv3_2out = tf.nn.relu(conv3_2out)

shape = int(np.prod(conv3_2out.get_shape()[1:]))
fcS3w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS3b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv3_2out, [-1, shape])
resultS3 = tf.nn.bias_add(tf.matmul(flat, fcS3w), fcS3b)
ansS3 = tf.nn.softmax(resultS3)

#4
sideFilter4_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias4_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv4_1out = tf.nn.conv2d(tf.nn.relu(conv2_2),sideFilter4_1, [1, 1, 1, 1], padding='SAME')
conv4_1out = tf.nn.bias_add(conv4_1out,sideBias4_1)

resS4 = conv1_1out + conv2_1out +conv3_1out + conv4_1out
resS4 = tf.nn.relu(resS4)

sideFilter4_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias4_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv4_2out = tf.nn.conv2d(resS4,sideFilter4_2, [1, 1, 1, 1], padding='SAME')
conv4_2out = tf.nn.bias_add(conv4_2out,sideBias4_2)
conv4_2out = tf.nn.relu(conv4_2out)

shape = int(np.prod(conv4_2out.get_shape()[1:]))
fcS4w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS4b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv4_2out, [-1, shape])
resultS4 = tf.nn.bias_add(tf.matmul(flat, fcS4w), fcS4b)
ansS4 = tf.nn.softmax(resultS4)
#5
sideFilter5_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias5_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv5_1out = tf.nn.conv2d(conv3_1,sideFilter5_1, [1, 1, 1, 1], padding='SAME')
conv5_1out = tf.nn.bias_add(conv5_1out,sideBias5_1)

resS5 = conv1_1out + conv2_1out + conv3_1out + conv4_1out + conv5_1out
resS5 = tf.nn.relu(resS5)

sideFilter5_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias5_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv5_2out = tf.nn.conv2d(resS5,sideFilter5_2, [1, 1, 1, 1], padding='SAME')
conv5_2out = tf.nn.bias_add(conv5_2out,sideBias5_2)
conv5_2out = tf.nn.relu(conv5_2out)

shape = int(np.prod(conv5_2out.get_shape()[1:]))
fcS5w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS5b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv5_2out, [-1, shape])
resultS5 = tf.nn.bias_add(tf.matmul(flat, fcS5w), fcS5b)
ansS5 = tf.nn.softmax(resultS5)

#6
sideFilter6_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias6_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv6_1out = tf.nn.conv2d(conv3_2,sideFilter6_1, [1, 1, 1, 1], padding='SAME')
conv6_1out = tf.nn.bias_add(conv6_1out,sideBias6_1)

resS6 = conv1_1out + conv2_1out + conv3_1out + conv4_1out + conv5_1out + conv6_1out
resS6 = tf.nn.relu(resS6)

sideFilter6_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias6_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv6_2out = tf.nn.conv2d(resS6,sideFilter6_2, [1, 1, 1, 1], padding='SAME')
conv6_2out = tf.nn.bias_add(conv6_2out,sideBias6_2)
conv6_2out = tf.nn.relu(conv6_2out)

shape = int(np.prod(conv6_2out.get_shape()[1:]))
fcS6w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS6b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv6_2out, [-1, shape])
resultS6 = tf.nn.bias_add(tf.matmul(flat, fcS6w), fcS6b)
ansS6 = tf.nn.softmax(resultS6)
#7
sideFilter7_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias7_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv7_1out = tf.nn.conv2d(conv4_1,sideFilter7_1, [1, 1, 1, 1], padding='SAME')
conv7_1out = tf.nn.bias_add(conv7_1out,sideBias7_1)

resS7 = conv1_1out + conv2_1out + conv3_1out + conv4_1out + conv5_1out + conv6_1out + conv7_1out
resS7 = tf.nn.relu(resS7)

sideFilter7_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias7_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv7_2out = tf.nn.conv2d(resS7,sideFilter7_2, [1, 1, 1, 1], padding='SAME')
conv7_2out = tf.nn.bias_add(conv7_2out,sideBias7_2)
conv7_2out = tf.nn.relu(conv7_2out)

shape = int(np.prod(conv7_2out.get_shape()[1:]))
fcS7w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS7b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv7_2out, [-1, shape])
resultS7 = tf.nn.bias_add(tf.matmul(flat, fcS7w), fcS7b)
ansS7 = tf.nn.softmax(resultS7)
#8
sideFilter8_1 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias8_1 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv8_1out = tf.nn.conv2d(conv4_2,sideFilter8_1, [1, 1, 1, 1], padding='SAME')
conv8_1out = tf.nn.bias_add(conv8_1out,sideBias8_1)

resS8 = conv1_1out + conv2_1out + conv3_1out + conv4_1out + conv5_1out + conv6_1out + conv7_1out + conv8_1out
resS8 = tf.nn.relu(resS8)

sideFilter8_2 = tf.Variable(tf.truncated_normal([3,3,128,128], dtype=tf.float32,stddev=1e-1),trainable=True)
sideBias8_2 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True)
conv8_2out = tf.nn.conv2d(resS8,sideFilter8_2, [1, 1, 1, 1], padding='SAME')
conv8_2out = tf.nn.bias_add(conv8_2out,sideBias8_2)
conv8_2out = tf.nn.relu(conv8_2out)

shape = int(np.prod(conv8_2out.get_shape()[1:]))
fcS8w = tf.Variable(tf.truncated_normal([shape, 9],dtype=tf.float32,stddev=1e-1),trainable=True)
fcS8b = tf.Variable(tf.constant(1.0, shape=[9], dtype=tf.float32),trainable=True)
flat = tf.reshape(conv8_2out, [-1, shape])
resultS8 = tf.nn.bias_add(tf.matmul(flat, fcS8w), fcS8b)
ansS8 = tf.nn.softmax(resultS8)


resultT = (resultS1 + resultS2 + resultS3+ resultS4 + resultS5 + resultS6 + resultS7 + resultS8)/8
ansT = (ansS1 + ansS2 + ansS3 + ansS4 + ansS5 + ansS6 + ansS7 + ansS8)/8

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS3)) + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS1)) + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS2)) + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS4)) + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS5)) + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS6)) +tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS7))+tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = ground_truth,logits = resultS8))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
      




data = scipy.io.loadmat('Indian_pines_corrected.mat')
label = scipy.io.loadmat('Indian_pines_gt.mat')
data = data['indian_pines_corrected']
data = np.pad(data,((2,2), (2,2),(0,0)),mode='constant', constant_values = 0)
label = label['indian_pines_gt']
label = np.pad(label,((2,2), (2,2)),mode='constant', constant_values = 0)


class1 = np.array(np.where(label==1))
class2 = np.array(np.where(label==2))
class3 = np.array(np.where(label==3))
class4 = np.array(np.where(label==4))
class5 = np.array(np.where(label==5))
class6 = np.array(np.where(label==6))
class7 = np.array(np.where(label==7))
class8 = np.array(np.where(label==8))
class9 = np.array(np.where(label==9))
class10 = np.array(np.where(label==10))
class11 = np.array(np.where(label==11))
class12 = np.array(np.where(label==12))
class13 = np.array(np.where(label==13))
class14 = np.array(np.where(label==14))
class15 = np.array(np.where(label==15))
class16 = np.array(np.where(label==16))

#tr1 = np.random.choice(range(46), 5, replace=False)
#tst1 = np.array(np.setdiff1d(range(46),tr1))
tr2 = np.random.choice(range(1428), 200, replace=False)
tst2 = np.array(np.setdiff1d(range(1428),tr2))
tr3 = np.random.choice(range(830), 200, replace=False)
tst3 = np.array(np.setdiff1d(range(830),tr3))
#tr4 = np.random.choice(range(237), 24, replace=False)
#tst4 = np.array(np.setdiff1d(range(237),tr4))
tr5 = np.random.choice(range(483),200, replace=False)
tst5 = np.array(np.setdiff1d(range(483),tr5))
tr6 = np.random.choice(range(730), 200, replace=False)
tst6 = np.array(np.setdiff1d(range(730),tr6))
#tr7 = np.random.choice(range(28), 3, replace=False)
#tst7 = np.array(np.setdiff1d(range(28),tr7))
tr8 = np.random.choice(range(478), 200, replace=False)
tst8 = np.array(np.setdiff1d(range(478),tr8))
#tr9 = np.random.choice(range(20), 2, replace=False)
#tst9 = np.array(np.setdiff1d(range(20),tr9))
tr10 = np.random.choice(range(972),200, replace=False)
tst10 = np.array(np.setdiff1d(range(972),tr10))
tr11 = np.random.choice(range(2455), 200, replace=False)
tst11 = np.array(np.setdiff1d(range(2455),tr11))
tr12 = np.random.choice(range(593),200, replace=False)
tst12 = np.array(np.setdiff1d(range(593),tr12))
#tr13 = np.random.choice(range(205), 21, replace=False)
#tst13 = np.array(np.setdiff1d(range(205),tr13))
tr14 = np.random.choice(range(1265),200, replace=False)
tst14 = np.array(np.setdiff1d(range(1265),tr14))
#tr15 = np.random.choice(range(386), 39, replace=False)
#tst15 = np.array(np.setdiff1d(range(386),tr15))
#tr16 = np.random.choice(range(93), 10, replace=False)
#tst16 = np.array(np.setdiff1d(range(93),tr16))

#training_data_index = np.transpose(class1[:,tr1])
#training_data_index = np.append(training_data_index,np.transpose(class2[:,tr2]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class3[:,tr3]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class4[:,tr4]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class5[:,tr5]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class6[:,tr6]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class7[:,tr7]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class8[:,tr8]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class9[:,tr9]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class10[:,tr10]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class11[:,tr11]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class12[:,tr12]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class13[:,tr13]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class14[:,tr14]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class15[:,tr15]),axis = 0)
#training_data_index = np.append(training_data_index,np.transpose(class16[:,tr16]),axis = 0)


training_data_index = np.transpose(class2[:,tr2])
training_data_index = np.append(training_data_index,np.transpose(class3[:,tr3]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class5[:,tr5]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class6[:,tr6]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class8[:,tr8]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class10[:,tr10]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class11[:,tr11]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class12[:,tr12]),axis = 0)
training_data_index = np.append(training_data_index,np.transpose(class14[:,tr14]),axis = 0)

#testing_data_index = np.transpose(class1[:,tst1])
#testing_data_index = np.append(testing_data_index,np.transpose(class2[:,tst2]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class3[:,tst3]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class4[:,tst4]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class5[:,tst5]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class6[:,tst6]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class7[:,tst7]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class8[:,tst8]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class9[:,tst9]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class10[:,tst10]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class11[:,tst11]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class12[:,tst12]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class13[:,tst13]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class14[:,tst14]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class15[:,tst15]),axis = 0)
#testing_data_index = np.append(testing_data_index,np.transpose(class16[:,tst16]),axis = 0)


testing_data_index = np.transpose(class2[:,tst2])
testing_data_index = np.append(testing_data_index,np.transpose(class3[:,tst3]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class5[:,tst5]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class6[:,tst6]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class8[:,tst8]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class10[:,tst10]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class11[:,tst11]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class12[:,tst12]),axis = 0)
testing_data_index = np.append(testing_data_index,np.transpose(class14[:,tst14]),axis = 0)

print(testing_data_index.shape)

np.savetxt('trainingIndianPines.txt', training_data_index)
np.savetxt('testingIndianPines.txt', testing_data_index)

training_data_index = np.loadtxt('trainingIndianPines.txt')
testing_data_index = np.loadtxt('testingIndianPines.txt')
training_data_index = training_data_index.astype(int)
testing_data_index = testing_data_index.astype(int)

accP = 0
accP2 = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess.run(init)
	#try:
	#	saver.restore(sess, "/tmp/model.ckpt")
	#except:
	#	print("no model")
	for k in range(1500):
		lossT = 0
		acc = 0
		acc2= 0
		for i in range(1):
			inp = data[training_data_index[(0),0]-2:training_data_index[(0),0]+3,training_data_index[(0),1]-2:training_data_index[(0),1]+3,:]
			inp = np.array([inp])
			encoding = np.zeros(9)
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==2):
				encoding[0] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==3):
				encoding[1] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==5):
				encoding[2] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==6):
				encoding[3] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==8):
				encoding[4] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==10):
				encoding[5] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==11):
				encoding[6] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==12):
				encoding[7] = 1
			if(label[training_data_index[(0),0],training_data_index[(0),1]]==14):
				encoding[8] = 1
			encoding = np.array([encoding])
			for j in range(1,1800):
				tempINP = data[training_data_index[j,0]-2:training_data_index[j,0]+3,training_data_index[j,1]-2:training_data_index[j,1]+3,:]
				tempINP = np.array([tempINP])
				tempECD = np.zeros(9)
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==2):
					tempECD[0] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==3):
					tempECD[1] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==5):
					tempECD[2] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==6):
					tempECD[3] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==8):
					tempECD[4] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==10):
					tempECD[5] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==11):
					tempECD[6] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==12):
					tempECD[7] = 1
				if(label[training_data_index[(j),0],training_data_index[(j),1]]==14):
					tempECD[8] = 1
				#print(tempECD)
				tempECD = np.array([tempECD])
				inp = np.append(inp,tempINP,axis = 0)
				encoding = np.append(encoding,tempECD,axis = 0)
			opt,lossP,ansP,final= sess.run([optimizer,loss,ansS8,ansT],feed_dict={input_patch:inp,ground_truth:encoding})
			lossT = lossT + lossP
		#for i in range(1030):
		#	inp = data[training_data_index[(i),0]-2:training_data_index[(i),0]+3,training_data_index[(i),1]-2:training_data_index[(i),1]+3,:]
		#	inp = np.array([inp])
		#	encoding = np.zeros(16)
		#	encoding[label[training_data_index[(i),0],training_data_index[(i),1]]-1] = 1
		#	encoding = np.array([encoding])
		#	opt,lossP,ansP,final= sess.run([optimizer,loss,ans,ansT],feed_dict={input_patch:inp,ground_truth:encoding})
		#	lossT = lossT + lossP
		#	#print(ansP[0,:].shape)
		#	if(np.argmax(ansP[0,:])==np.argmax(encoding[0,:])):
		#		acc = acc + 1
		#	if(np.argmax(final[0,:])==np.argmax(encoding[0,:])):
		#		acc2 = acc2 + 1
			print(lossT)
			for p in range(1800):
				if(np.argmax(ansP[p,:])==np.argmax(encoding[p,:])):
					acc = acc + 1
				if(np.argmax(final[p,:])==np.argmax(encoding[p,:])):
					acc2 = acc2 + 1
			print(acc/1800,accP/7434)
			print(acc2/1800,accP2/7434)
			if(k%15 == 0):
				save_path = saver.save(sess, "/tmp/model.ckpt")
				accP = 0
				accP2 = 0
				#for l in range(1):
				#	inp = data[testing_data_index[0,0]-4:testing_data_index[0,0]+5,testing_data_index[0,1]-4:testing_data_index[0,1]+5,:]
				#	inp = np.array([inp])
				#	encoding = np.zeros(16)
				#	encoding[label[testing_data_index[0,0],testing_data_index[0,1]]-1] = 1
				#	encoding = np.array([encoding])
				#	for n in range(1,9219):
				#		tempINP = data[testing_data_index[n,0]-4:testing_data_index[n,0]+5,testing_data_index[n,1]-4:testing_data_index[n,1]+5,:]
				#		tempINP = np.array([tempINP])
				#		tempECD = np.zeros(16)
				#		tempECD[label[testing_data_index[n,0],testing_data_index[n,1]]-1] = 1
				#		tempECD = np.array([tempECD])
				#		inp = np.append(inp,tempINP,axis = 0)
				#		encoding = np.append(encoding,tempECD,axis = 0)
				#	ansPred,finalPred= sess.run([ans,ansT],feed_dict={input_patch:inp,ground_truth:encoding})
				#	ansPred = np.array(ansPred)
				#	for q in range(9219):
				#		if(np.argmax(ansPred[q,:])==np.argmax(encoding[q,:])):
				#			accP = accP + 1
				#		if(np.argmax(finalPred[q,:])==np.argmax(encoding[q,:])):
				#			accP2 = accP2 + 1
				for l in range(7434):
					inp = data[testing_data_index[l,0]-2:testing_data_index[l,0]+3,testing_data_index[l,1]-2:testing_data_index[l,1]+3,:]
					inp = np.array([inp])
					encoding = np.zeros(9)
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==2):
						encoding[0] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==3):
						encoding[1] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==5):
						encoding[2] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==6):
						encoding[3] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==8):
						encoding[4] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==10):
						encoding[5] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==11):
						encoding[6] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==12):
						encoding[7] = 1
					if(label[testing_data_index[(l),0],testing_data_index[(l),1]]==14):
						encoding[8] = 1
					encoding = np.array([encoding])
					ansPred,finalPred= sess.run([ansS8,ansT],feed_dict={input_patch:inp,ground_truth:encoding})
					if(np.argmax(ansPred)==np.argmax(encoding)):
						accP = accP + 1
					if(np.argmax(finalPred)==np.argmax(encoding)):
						accP2 = accP2 + 1