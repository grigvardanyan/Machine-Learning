import tensorflow as tf 
import numpy as np
#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

#Store the MNIST data
mnist = input_data.read_data_sets("mnist_data/",one_hot=True)

training_digits,training_labels = mnist.train.next_batch(30000)

test_digits, test_labels = mnist.test.next_batch(1000)

traiining_digits_pl = tf.placeholder("float",[None,784])

test_digit_pl = tf.placeholder("float",[784])

#NN calculation using L1 distance
L1_distance = tf.abs(tf.add(traiining_digits_pl,tf.negative(test_digit_pl)))

distance = tf.reduce_sum(L1_distance,axis=1)

#Prediction:Get min distance index (NN)
pred = tf.arg_min(distance,0)

accuracy = 0.
#Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_digits)):
        #Get NN
        nn_index = sess.run(pred,feed_dict={traiining_digits_pl:training_digits,test_digit_pl:test_digits[i,:]})
        #Get NN class label and compare it to its true label
        KNN_result = "Test - " + str(i) + "  Prediction: " + str(np.argmax(training_labels[nn_index])) + "  True Label: " + str(np.argmax(test_labels[i]))
        print(KNN_result)
        #Calculate accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)
    print("Done!")
    print("Accuracy", accuracy)  