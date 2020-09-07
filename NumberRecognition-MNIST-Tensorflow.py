#
# @author erensayar 
#

#Kendi çizimimizi okuyabilmek için OpenCV kütüphanesi import edilmiştir.
import cv2
#Array işlemleri için numpy import edilmiştir.
import numpy as np
import tensorflow as tf

#Dataset tensorflow paketinden import edilir.
from tensorflow.examples.tutorials.mnist import input_data
#Dataset okunur
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Placeholder'lar tanımlanır
#Eğitim sürecinde karşılaştırmalar yapılacaktır.
#Bu yüzden placeholder eklendi, geçici değerler gibi düşünülebilir.
x = tf.placeholder(tf.float32, [None, 784])#Softmax regresyonu için
y_ = tf.placeholder(tf.float32, [None, 10])#Çapraz entropi için(accuracy)

#Ağırlıklar ve sapmaların ayarlanması için değişkenler.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Çıktı katmanı ağı oluşturulur
#Softmax regresyonu kullanılır
y = tf.nn.softmax(tf.matmul(x, W) + b)

#(Loss Fonksiyonu)
#Çapraz Entropi algoritması.[Döküman içerisinde olan L(a,y) loss fonskiyonu.]
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Geri yayılım algoritması ve gradyan inişi(0,5 oranı ile) gerçekleşir(Ağırlıklar optimize edilir)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Model başlatılır
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Eğitim işlemi gerçekleşir.(1000 kez)
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
#Modelin ne kadar iyi olduğu değerlendirilir.
#Accuracy hesaplanır
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Accuracy oranı çıktısı verilir.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Kendi çizimim OpenCV aracılığıyla gri renk düzeyinde okunur.
#Tahmin gerçekleştirilir, sonuç çıktısı verilir.
cizim = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(cv2.imread('E:\EREN\CODES\Python\MNIST_Tensorflow\Rakam0.png',cv2.IMREAD_GRAYSCALE)))
sonuc = sess.run(tf.argmax(y, 1), feed_dict={x: [cizim]})
print(sonuc)

