import numpy as np
import tensorflow as tf
from tensorlayer import *
import os


SAVE_PATH = '/Models/IrisCompare/model.ckpt'
SUMMARY_FILEPATH = './Models/SeaCliff/Summaries/'
FILEPATH = './Data/CASIA/Training/'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels_fine'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images_fine'
TEST_LABEL_SAVE = FILEPATH + 'test_labels_fine'
TEST_INPUT_SAVE = FILEPATH + 'test_images_fine'
ITERATIONS = 1000000
CONVOLUTIONS = []
IMAGE_SIZE = 128,128,2
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
RESTORE = False
WHEN_SAVE = 1000

def decode_image(image):
    image = tf.decode_raw(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]])
    return image

def decode_label(label):
    return  tf.decode_raw(label, tf.uint8)

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE, 1).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def build_model(x, y):
    conv_pointer = InputLayer(x, name= 'f_disc_inputs')
    for i,v in enumerate(CONVOLUTIONS):
        if i < (len(CONVOLUTIONS) - 1):#if its the last layer choose tanh
            act = tf.nn.relu
        else:
            act = tf.nn.tanh
        if v < 0:#if convolution is negative use maxpooling
            v *= -1
            curr_layer = Conv2d(BatchNormLayer(conv_pointers,
                act=act,is_train=True ,name=
                'batch_norm%s'%(i)),
                v, (5, 5),strides = (1,1), name=
                'conv1_%s'%(i))
            curr_layer = MaxPool2d(curr_layer, filter_size = (2,2), strides = (2,2), name = 'pool_%s'%(i))
            conv_pointer = curr_layer
        else:
            curr_layer = Conv2d(BatchNormLayer(conv_pointers[-1],
            act=act,is_train=True ,name=
            'batch_norm%s'%(i)),
                v, (5, 5),strides = (1,1), name=
                'conv1_%s'%(i))
            conv_pointer= curr_layer
    _, pm_width, pm_height, _ = conv_pointer.outputs.get_shape()
    conv_pointer = MeanPool2d(conv_pointer, filter_size = (pm_width, pm_width), strides = (pm_width, pm_width), name = 'Final_Pool')
    logits = FlattenLayer(conv_pointer).outputs
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    final_guess = tf.round(tf.sigmoid(logits))
    accuracy_summary = tf.summary.scalar('Accuracy',tf.reduce_mean(tf.cast(tf.equal(flat_labels, final_guess), tf.float32)))
    cross_entropy_summary = tf.summary.scalar('Cross Entropy', tf.reduce_mean(cross_entropy))
        flat_labels = tf.cast(flat_labels, tf.float64)
    final_guess = tf.cast(final_guess, tf.float64)
    TP = tf.count_nonzero(final_guess * y, dtype=tf.float32)
    TN = tf.count_nonzero((final_guess - 1) * (y - 1), dtype=tf.float32)
    FP = tf.count_nonzero(final_guess * (y - 1), dtype=tf.float32)
    FN = tf.count_nonzero((final_guess - 1) * y, dtype=tf.float32)
    true_positive = tf.divide(TP, TP + FP)
    true_negative = tf.divide(TN, TN + FN)
    true_positive_summary =tf.summary.scalar('True Positive',true_positive)
    true_negative_summary =tf.summary.scalar('True Negative',true_negative)
    total_summary = tf.sumary.merge([accuracy_summary,cross_entropy_summary,true_positive_summary,true_negative_summary])
    return train_step, total_summary


if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    # test_ship = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    # test_iterator = test_mnist.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    # test_input = test_iterator.get_next()
    sess.run([train_iterator.initializer])
    input_summary, train_step = build_model(train_input, train_label)

    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())
    #
    saver_perm = tf.train.Saver()
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, SAVE_PATH)
    else:
        print('SAVE')
        saver_perm.save(sess, SAVE_PATH)


    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    for i in range(ITERATIONS):
        input_summary_ex, _= sess.run([input_summary, train_step])
        train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_SAVE:
            saver_perm.save(sess, SAVE_PATH)
