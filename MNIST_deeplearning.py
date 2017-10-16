__author__ = 'linh'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x,[-1, 28, 28, 1]) # 2 cai giua la weight vs height cua image, 1 la mau, anh den trang

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



def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
#first layer
#some questions:
#1. tsao input la 1 channel --> check shape of x
#2. tsao layer 1 la 32, layer 2 la 64 --> tuong phai nho lai chu nhi
# --> ah hieu r, W phai to len thi size output se nho lai, no nhan vs nhau ma
# --> lai them vde la tsao W phai nhieu chieu the, tuong 2 chieu la ok chu nhi --> in ra
#3. check tdung cua reshape
#4. check max_pool. Van cha hieu sao no phai lam the

        W_conv = weight_variable([5,5,channels_in, channels_out], name="W") #2 first ones are patch size, 1 is input channels, 32 is output channels
        b_conv = bias_variable([channels_out], name="b")

        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)
        return h_pool


def fc_layer(input, channels_in, channels_out, name="fc"):
#Densely connected layer
#h image la 7x7 -> nho check lai
# nhung tsao lai 1024 nhi? --> cha lquan mie j -->kieu flat ra thoi -> ti thay so khac xem tn
# k hieu sao flat lai dung sau pool --> check lai dimension
    with tf.name_scope(name):

        W_fc = weight_variable([channels_in, channels_out], name="W")
        b_fc = bias_variable([channels_out], name="b")

        h_pool_flat = tf.reshape(input, [-1, channels_in])
        h_fc = tf.matmul(h_pool_flat, W_fc) + b_fc
        return h_fc


conv1 = conv_layer(x_image, 1, 32, "conv1")
conv2 = conv_layer(conv1, 32, 64, "conv2")

flattened = tf.reshape(conv2, [-1, 7*7*64])
fc_1 = fc_layer(flattened, 7*7*64, 1024, "fc1")
keep_prob = tf.placeholder(tf.float32)
fc_1_drop = tf.nn.dropout(tf.nn.relu(fc_1))
y_conv = fc_layer(fc_1_drop, 1024, 10, "fc2")



#train
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/2')
    writer.add_graph(sess.graph)


    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

