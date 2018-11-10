#!/usr/bin/env python
# coding: utf-8

# ### Deep Learning Up and Running
# #### By MiMoTrix
# ##### MNIST with FullyConnected Network

# In[21]:


import tensorflow as tf


# In[23]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# ##### Initial Values

# In[24]:


learning_rate = 0.01
epochs = 1000
batch_size = 64


# ##### Input Placeholders

# In[25]:


X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name="Inputs")
Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Lables")


# ##### Layer One Weight and Bias Variables Initialize

# In[26]:


with tf.name_scope(name="HiddenLayerOne"):
    HiddenLayerOneWeight = tf.Variable(tf.truncated_normal([784, 500], 0, 0.1), name="HiddenLayerOneWeight")
    HiddenLayerOneBias = tf.Variable(tf.zeros([500]), name="HiddenLayerOneBias")
    HiddenLayerOneOutput = tf.nn.relu(tf.matmul(X, HiddenLayerOneWeight) + HiddenLayerOneBias)


# ##### Layer Two Weight and Bias Variables Initialize

# In[27]:


with tf.name_scope(name="HiddenLayerTwo"):
    HiddenLayerTwoWeight = tf.Variable(tf.truncated_normal([500, 500], 0, 0.1), name="HiddenLayerTwoWeight")
    HiddenLayerTwoBias = tf.Variable(tf.zeros([500]), name="HiddenLayerTwoBias")
    HiddenLayerTwoOutput = tf.nn.relu(tf.matmul(HiddenLayerOneOutput, HiddenLayerTwoWeight) + HiddenLayerTwoBias)


# ##### Output Layer Weight and Bias Variables Initialize

# In[28]:


with tf.name_scope("OutputLayer"):
    OutputLayerWeight = tf.Variable(tf.truncated_normal([500, 10], 0, 0.1), name="OutputLayerWeight")
    OutputLayerBias = tf.Variable(tf.zeros([10]), name="OutputLayerBias")
    OutputLayerOutput = tf.nn.softmax(tf.matmul(HiddenLayerTwoOutput, OutputLayerWeight) + OutputLayerBias)


# ##### Cross Entropy, Loss Function, Accuaracy, etc.

# In[29]:


with tf.name_scope("Loss"):
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=OutputLayerOutput))
with tf.name_scope("Training"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(crossEntropy)
with tf.name_scope("Accuracy"):
    prediction = tf.equal(tf.argmax(OutputLayerOutput, 1), tf.argmax(Y, 1))
    accOp = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32), name="Accuracy")


# ##### Session

# In[30]:


tf.summary.scalar("CrossEntropy", crossEntropy)
tf.summary.scalar("TrainingAccuracy", accOp)

merge = tf.summary.merge_all()
filewriter = tf.summary.FileWriter("./graphs")
sess = tf.Session()
filewriter.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())


# ##### Train

# In[31]:


acc = tf.summary.scalar("AccuracyValidation", accOp)
for iteration in range(epochs):
    X_Batch, Y_Batch = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={X: X_Batch, Y: Y_Batch})
    if iteration % 50 == 0:
        validation = (sess.run(merge, feed_dict={X: X_Batch, Y: Y_Batch}))
        filewriter.add_summary(validation, iteration)
        X_Batch_Validation, Y_Batch_Validation = mnist.validation.next_batch(batch_size)
        validation = (sess.run(acc, feed_dict={X: X_Batch_Validation, Y: Y_Batch_Validation}))
        filewriter.add_summary(validation, iteration)


# In[13]:


outputValidation = sess.run(OutputLayerOutput, feed_dict={X: mnist.test.images})
[totalAcc] = sess.run([accOp], feed_dict={OutputLayerOutput: outputValidation, Y: mnist.test.labels})
print("Total Accuracy is " + str(totalAcc))
sess.close()
filewriter.close()


# In[ ]:




