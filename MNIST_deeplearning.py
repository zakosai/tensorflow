__author__ = 'linh'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")



#first layer
#some questions:
#1. tsao input la 1 channel --> check shape of x
#2. tsao layer 1 la 32, layer 2 la 64 --> tuong phai nho lai chu nhi
# --> ah hieu r, W phai to len thi size output se nho lai, no nhan vs nhau ma
# --> lai them vde la tsao W phai nhieu chieu the, tuong 2 chieu la ok chu nhi --> in ra
#3. check tdung cua reshape
#4. check max_pool. Van cha hieu sao no phai lam the
W_conv1 = weight_variable([5,5,1,32]) #2 first ones are patch size, 1 is input channels, 32 is output channels
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1, 28, 28, 1]) # 2 cai giua la weight vs height cua image, 1 la mau, anh den trang
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#second layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Densely connected layer
#h image la 7x7 -> nho check lai
# nhung tsao lai 1024 nhi? --> cha lquan mie j -->kieu flat ra thoi -> ti thay so khac xem tn
# k hieu sao flat lai dung sau pool --> check lai dimension


W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



#Dropout -> De giam overfiting -> ma van chua biet lsao de giam
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Readout Layer -> layer cuoi, dung softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    file_write = tf.summary.FileWriter('logs/', sess.graph)
    for i in range(1):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        h_c, h_p = sess.run([h_conv1, h_pool1],feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        # h_c = h_conv1.eval()
        # h_p = h_pool1.eval()
        print(h_c.shape)
        print(h_p.shape)

    print('test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

