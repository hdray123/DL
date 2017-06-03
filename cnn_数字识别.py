import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def Weight(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def baisa(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def polmax_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs=tf.placeholder(tf.float32,[None,784])/255
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_images=tf.reshape(xs,[-1,28,28,1])#[-1,28,28,1] -1为数量不限，28x28为图像大小 ，1 为图像深度

#layer_1
w_1=Weight([5,5,1,32])
b_1=baisa([32])
h_1=conv2d(x_images,w_1)+b_1
h_pol_1=polmax_2x2(h_1)     #14x14x32
#layer_2
w_2=Weight([5,5,32,64])
b_2=baisa([64])
h_2=conv2d(h_pol_1,w_2)+b_2
h_pol_2=polmax_2x2(h_2)  #7x7x64
#layer_3
#w_5=Weight([3,3,64,128])
#b_5=baisa([128])
#h_5=conv2d(h_pol_2,w_5)+b_5 #7x7x128
#fct1
w_3=Weight([7*7*64,1024])
b_3=baisa([1024])
h_pol_2_fat=tf.reshape(h_pol_2,[-1,7*7*64])
h_3=tf.nn.relu(tf.matmul(h_pol_2_fat,w_3)+b_3)
h_3_dropout=tf.nn.dropout(h_3,keep_prob)

#fct2
w_4=Weight([1024,10])
b_4=baisa([10])
prediction=tf.nn.softmax(tf.matmul(h_3_dropout,w_4)+b_4)

#loss
coss_enteropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction)
                                            ,reduction_indices=[1]))


#train

train_step=tf.train.AdamOptimizer(1e-4).minimize(coss_enteropy)

initin=tf.global_variables_initializer()
sess=tf.Session()
sess.run(initin)
for i in range(5000):
    x_batch,y_batch=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:x_batch,ys:y_batch,keep_prob:0.5})
    if i%200==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

