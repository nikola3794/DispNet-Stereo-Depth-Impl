import os
from os import walk
import time
import re
import numpy as np
import sys
import tensorflow as tf
import imageio
import scipy.misc
import random



# read disparity data from .pfm file
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


# write disparity data to .pfm file
def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)

    return


#Convolution layer
def conv(x, name, shape, stride, relu, batch_norm):
    with tf.variable_scope(name):
        #Weights
        initial_w = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name='W', shape=shape, initializer=initial_w)

        #Bias
        initial_b = tf.constant(value=0.0, shape=[shape[3]])
        b = tf.get_variable(name='b', initializer=initial_b)

        #Convolution
        h_conv = tf.nn.conv2d(input=x, filter=W, strides=stride, padding='SAME') + b

        if batch_norm:
            #Batch normalization
            h_conv = tf.contrib.layers.batch_norm(h_conv)
        if relu:
            #Apply ReLu activation
            h_conv = tf.nn.relu(h_conv)

        return h_conv

#Deconvolution layer
def deconv(x, name, shape, stride, relu, batch_norm):
    with tf.variable_scope(name):
        #Weights
        initial_w = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable(name='W', shape=shape, initializer=initial_w)

        #Bias
        initial_b = tf.constant(value=0.0, shape=[shape[2]])
        b = tf.get_variable(name='b', initializer=initial_b)

        #New shape from input shape
        input_shape = tf.shape(x)
        deconv_shape = tf.stack([input_shape[0], input_shape[1] * 2, input_shape[2] * 2, shape[2]])
        h_deconv = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=deconv_shape, strides=stride) + b

        if batch_norm:
            # Batch normalization
            h_conv = tf.contrib.layers.batch_norm(h_deconv)
        if relu:
            # Apply ReLu activation
            h_conv = tf.nn.relu(h_deconv)

        return h_conv


#2x2 max pooling with 2x2 stride and padding
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''
train_file_list = []
test_file_list = []

image_data_path = 'D:\FlyingThings3D_dataset\FlyingThings3D_subset'
for (dirpath, dirnames, filenames) in walk(image_data_path):
    print(filenames)
'''

# load m data samples (image pairs with disparity) from file list to tensors
def LoadSamples(m, file_batch, image_height, image_width, image_downsize):

    x_batch = np.empty([m, image_height // image_downsize, image_width // image_downsize, 6])
    y_batch = np.empty([m, image_height // image_downsize, image_width // image_downsize, 1])

    for i in range(m):
        #left image
        img_left = scipy.misc.imread(file_batch[i][0])
        x_batch[i, :, :, 0 : 3] = img_left[0:image_height // image_downsize, 0:image_width // image_downsize]
        #imageio.imwrite('img_left.png', x_batch[i, :, :, 0 : 3])

        #right image
        img_right = scipy.misc.imread(file_batch[i][1])
        x_batch[i, :, :, 3 : 6] = img_right[0:image_height // image_downsize, 0:image_width // image_downsize]
        #imageio.imwrite('img_right.png', x_batch[i, :, :, 3 : 6])

        # disparity image
        img_disp = readPFM(file_batch[i][2])[0]
        y_batch[i, :, :, 0] = img_disp[0:image_height // image_downsize, 0:image_width // image_downsize]
        #imageio.imwrite('disp_left.png', y_batch[i, :, :, 0])


    return x_batch, y_batch

#Original size w = 960, h = 540
image_width = 768
image_height = 384
image_downsize = 1


batch_norm_conv = True
batch_norm_deconv = True



with tf.variable_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, image_height // image_downsize, image_width // image_downsize, 6])
    y = tf.placeholder(dtype=tf.float32, shape=[None, image_height // image_downsize, image_width // image_downsize, 1])
with tf.variable_scope('learning'):
    #Weights for 6 losses
    w_loss = tf.placeholder(dtype=tf.float32, shape=[6])
    #Learning rate
    lr = tf.placeholder(dtype=tf.float32)


#Constructing a CNN
h_conv1 = conv(x=x, name='conv1', shape=[7, 7, 6, 64], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)

h_conv2 = conv(x=h_conv1, name='conv2', shape=[5, 5, 64, 128], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)

h_conv3a = conv(x=h_conv2, name='conv3a', shape=[5, 5, 128, 256], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)
h_conv3b = conv(x=h_conv3a, name='conv3b', shape=[3, 3, 256, 256], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)

h_conv4a = conv(x=h_conv3b, name='conv4a', shape=[3, 3, 256, 512], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)
h_conv4b = conv(x=h_conv4a, name='conv4b', shape=[3, 3, 512, 512], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)

h_conv5a = conv(x=h_conv4b, name='conv5a', shape=[3, 3, 512, 512], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)
h_conv5b = conv(x=h_conv5a, name='conv5b', shape=[3, 3, 512, 512], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)

h_conv6a = conv(x=h_conv5b, name='conv6a', shape=[3, 3, 512, 1024], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_conv)
h_conv6b = conv(x=h_conv6a, name='conv6b', shape=[3, 3, 1024, 1024], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr6 = conv(x=h_conv6b, name='pr6', shape=[3, 3, 1024, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr6_up = deconv(x=h_pr6, name='pr6up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv5 = deconv(x=h_conv6b, name='upconv5', shape=[4, 4, 512, 1024], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input5 = tf.concat([h_upconv5, h_conv5b, h_pr6_up], axis=3, name='input5')
h_iconv5 = conv(x=input5, name='iconv5', shape=[3, 3, 1025, 512], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr5 = conv(x=h_iconv5, name='pr5', shape=[3, 3, 512, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr5_up = deconv(x=h_pr5, name='pr5up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv4 = deconv(x=h_iconv5, name='upconv4', shape=[4, 4, 256, 512], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input4 = tf.concat([h_upconv4, h_conv4b, h_pr5_up], axis=3, name='input4')
h_iconv4 = conv(x=input4, name='iconv4', shape=[3, 3, 769, 256], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr4 = conv(x=h_iconv4, name='pr4', shape=[3, 3, 256, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr4_up = deconv(x=h_pr4, name='pr4up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv3 = deconv(x=h_iconv4, name='upconv3', shape=[4, 4, 128, 256], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input3 = tf.concat([h_upconv3, h_conv3b, h_pr4_up], axis=3, name='input3')
h_iconv3 = conv(x=input3, name='iconv3', shape=[3, 3, 385, 128], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr3 = conv(x=h_iconv3, name='pr3', shape=[3, 3, 128, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr3_up = deconv(x=h_pr3, name='pr3up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv2 = deconv(x=h_iconv3, name='upconv2', shape=[4, 4, 64, 128], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input2 = tf.concat([h_upconv2, h_conv2, h_pr3_up], axis=3, name='input2')
h_iconv2 = conv(x=input2, name='iconv2', shape=[3, 3, 193, 64], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr2 = conv(x=h_iconv2, name='pr2', shape=[3, 3, 64, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr2_up = deconv(x=h_pr2, name='pr2up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv1 = deconv(x=h_iconv2, name='upconv1', shape=[4, 4, 32, 64], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input1 = tf.concat([h_upconv1, h_conv1, h_pr2_up], axis=3, name='input1')
h_iconv1 = conv(x=input1, name='iconv1', shape=[3, 3, 97, 32], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr1 = conv(x=h_iconv1, name='pr1', shape=[3, 3, 32, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)
h_pr1_up = deconv(x=h_pr1, name='pr1up', shape=[4, 4, 1, 1], stride=[1, 2, 2, 1], relu=False, batch_norm=batch_norm_deconv)

h_upconv0 = deconv(x=h_iconv1, name='upconv0', shape=[4, 4, 16, 32], stride=[1, 2, 2, 1], relu=True, batch_norm=batch_norm_deconv)
input0 = tf.concat([h_upconv0, h_pr1_up], axis=3, name='input0')
h_iconv0 = conv(x=input0, name='iconv0', shape=[3, 3, 17, 16], stride=[1, 1, 1, 1], relu=True, batch_norm=batch_norm_conv)
h_pr0 = conv(x=h_iconv0, name='pr0', shape=[3, 3, 16, 1], stride=[1, 1, 1, 1], relu=False, batch_norm=False)

with tf.variable_scope('loss'):
    #Projecting the label onto a lower resolution
    y_resize1 = tf.image.resize_images(y, [image_height // 2, image_width // 2])
    y_resize2 = tf.image.resize_images(y, [image_height // 4, image_width // 4])
    y_resize3 = tf.image.resize_images(y, [image_height // 8, image_width // 8])
    y_resize4 = tf.image.resize_images(y, [image_height // 16, image_width // 16])
    y_resize5 = tf.image.resize_images(y, [image_height // 32, image_width // 32])
    y_resize6 = tf.image.resize_images(y, [image_height // 64, image_width // 64])

    #Computing all loss functions
    loss0 = tf.sqrt(tf.reduce_mean(tf.square(h_pr0 - y)), name='loss0')
    #loss1 = tf.sqrt(tf.reduce_mean(tf.square(h_pr1 - y_resize1)), name='loss1')
    loss2 = tf.sqrt(tf.reduce_mean(tf.square(h_pr2 - y_resize2)), name='loss2')
    loss3 = tf.sqrt(tf.reduce_mean(tf.square(h_pr3 - y_resize3)), name='loss3')
    loss4 = tf.sqrt(tf.reduce_mean(tf.square(h_pr4 - y_resize4)), name='loss4')
    loss5 = tf.sqrt(tf.reduce_mean(tf.square(h_pr5 - y_resize5)), name='loss5')
    loss6 = tf.sqrt(tf.reduce_mean(tf.square(h_pr6 - y_resize6)), name='loss6')

    tmp0 = tf.multiply(loss0, w_loss[0], name='tmp0')
    tmp2 = tf.multiply(loss2, w_loss[1], name='tmp2')
    tmp3 = tf.multiply(loss3, w_loss[2], name='tmp3')
    tmp4 = tf.multiply(loss4, w_loss[3], name='tmp4')
    tmp5 = tf.multiply(loss5, w_loss[4], name='tmp5')
    tmp6 = tf.multiply(loss6, w_loss[5], name='tmp6')

    loss = tf.add(tmp0, tf.add(tmp2, tf.add(tmp3, tf.add(tmp4, tf.add(tmp5, tmp6)))), name='loss')

    # create loss summaries for training evaluation
    tf.summary.scalar('s_loss_total', loss)
    tf.summary.scalar('s_loss0', loss0)
    tf.summary.scalar('s_loss2', loss2)
    tf.summary.scalar('s_loss3', loss3)
    tf.summary.scalar('s_loss3', loss4)
    tf.summary.scalar('s_loss4', loss5)
    tf.summary.scalar('s_loss6', loss6)

    summary_op = tf.summary.merge_all()

with tf.variable_scope('train'):
    #Adam optimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(loss, name="adam_train_step")


'''
DATA LOADING
'''
train_file_list = []
test_file_list = []

image_data_path = r'D:\FlyingThings3D_dataset\FlyingThings3D_subset_img'
disparity_data_path = r'D:\FlyingThings3D_dataset\FlyingThings3D_subset_disp'
for (dirpath, dirnames, filenames) in walk(image_data_path):
    # print(dirpath)

    for file_name in filenames:
        file_path = dirpath + '\\' + file_name
        if 'left' in file_path.lower():
            left_image_path = file_path
            right_image_path = left_image_path.replace('left', 'right')

            disparity_image_path = left_image_path.replace(image_data_path, disparity_data_path)
            disparity_image_path = disparity_image_path.replace(r'.png', r'.pfm')


            if 'train' in dirpath.lower():
                train_file_list.extend([[left_image_path, right_image_path, disparity_image_path]])
            elif 'val' in dirpath.lower():
                test_file_list.extend([[left_image_path, right_image_path, disparity_image_path]])

train_file_len = len(train_file_list)
test_file_len = len(test_file_list)
#print(train_file_len)
#print(test_file_len)

'''
TRAINING INITIALIZATION
'''
new_dir = os.getcwd() + '\\test_images_fixed'
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

new_dir = os.getcwd() + '\\test_images_fixed'
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)


with tf.Session() as sess:
    #Initialize the model saver and summary writer
    saver = tf.train.Saver(max_to_keep=1)
    train_writer = tf.summary.FileWriter('.\\log\\train', sess.graph)
    test_writer = tf.summary.FileWriter('.\\log\\test', sess.graph)

    test_model = True

    restore = True
    if restore:
        saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + '\\model_save\\'))
        print('Models parameters are loaded.')
        # Load the epoch counter
        epoch = 12
        # Load the step (step-training on one batch) counter
        step = 75000
        # Load the loss weights
        loss_weight = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # Initialize all the models variables
        sess.run(tf.global_variables_initializer())
        print('Models parameters randomly initialized.')
        # Initialize the epoch counter
        epoch = 0
        # Initialize the step (step-training on one batch) counter
        step = 0
        # Initialize the loss weights
        loss_weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    #Initialize the batches position in the list of training data files
    batch_size = 4
    batch_start = 0
    batch_end = batch_start + batch_size
    batch_test_fixed = 10
    #TODO Save the dataset ordering and positions on exit, and load them on startup

    #Load fixed test examples for a visual inspection during training
    x_test_fixed, y_test_fixed = LoadSamples(batch_test_fixed, test_file_list, image_height, image_width, image_downsize)

    #Create the directories where the fixed test images will be stored
    for i in range(batch_test_fixed):
        new_dir = os.getcwd() + '\\test_images_fixed\\' + str(i)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
            #[Fixed test images]Save the original image along with the ground truth disparity
            imageio.imwrite(new_dir + '\\TRUEdisp__' + str(step) + '__' + str(i) + '.png', y_test_fixed[i][:, :, 0])
            imageio.imwrite(new_dir + '\\img__' + str(step) + '__' + str(i) + '.png', x_test_fixed[i][:, :, 0:3])

    #Randomly reorder the training&test set
    if not test_model:
        np.random.shuffle(test_file_list)
        np.random.shuffle(train_file_list)
    #Load random test examples for a visual inspection during training
    x_test_batch, y_test_batch = LoadSamples(batch_size, test_file_list, image_height, image_width, image_downsize)

    #Initialize the max number of epochs
    epoch_max = 150
    #How much steps until a model save
    step_save_model = 1000
    #How much steps until some images are tested
    step_test_images = 1000
    #How much steps until test loss is calculated and logged
    step_test_loss = 50
    #How much steps until train loss is logged
    step_train_loss = 10

    #Initialize the learning rate
    learning_rate = (1e-4)*0.5

    if test_model == True:
        test_acc = []
        batch_start_test = 0
        batch_end_test = batch_start_test + batch_size

    while epoch < epoch_max:
        if test_model == True:
            for i in range(1000):
                tmp_list = test_file_list[batch_start_test:batch_end_test]
                x_test_batch, y_test_batch = LoadSamples(batch_size, tmp_list, image_height, image_width, image_downsize)
                feed_dict_test = {x: x_test_batch, y: y_test_batch, w_loss: loss_weight, lr: learning_rate}
                tmp_loss = sess.run(loss, feed_dict=feed_dict_test)
                test_acc.append(tmp_loss)

                batch_start_test = batch_end_test
                batch_end_test = batch_start_test + batch_size
                print(i+1)
            print()
            print('Loss on the test set: {}'.format(np.mean(test_acc)))
            exit(37)

        #Increment the step counter
        step += 1

        #Load the training batch into a 4d numpy array
        tmp_list = train_file_list[batch_start:batch_end]
        x_train_batch, y_train_batch = LoadSamples(batch_size, tmp_list, image_height, image_width, image_downsize)

        #Prepare the batch positions in the list for the next epoch
        if (batch_end + batch_size) > train_file_len:
            # If we passed through the whole dataset, shuffle it and start again
            # (this may ignore a few examples at the end of the current dataset shuffle ordering)
            batch_start = 0
            batch_end = batch_start + batch_size
            np.random.shuffle(train_file_list)

            #Increment the number of epochs
            epoch += 1
        else:
            batch_start = batch_end
            batch_end = batch_start + batch_size

        #Prepare the training batch feed dictionary
        feed_dict_train = {x: x_train_batch, y: y_train_batch, w_loss: loss_weight, lr: learning_rate}

        #Run one training step with the current batch
        #train_step.run(feed_dict=feed_dict_train)

        #Compute the loss on the current training batch
        _, train_loss, summary = sess.run([train_step, loss, summary_op], feed_dict=feed_dict_train)
        print("Step: {} ...Epoch: {} ...Batch train loss: {}".format(step, epoch, train_loss))
        if step % step_train_loss == 0:
            # Log the loss values for plotting in tensorboard
            train_writer.add_summary(summary, step)
            #train_writer.flush()

        if step % step_test_loss == 0:
            feed_dict_test = {x: x_test_batch, y: y_test_batch, w_loss: loss_weight, lr: learning_rate}
            test_loss, summary = sess.run([loss, summary_op], feed_dict=feed_dict_test)
            test_writer.add_summary(summary, step)
            #test_writer.flush()

        #After 'step_test_images' number of steps, run 'batch_size' random test images through the CNN and save the results
        #Also, every 'step_test_images' number of steps, run 'batch_test_fixed' fixed test images through the CNN and save the results
        if step % step_test_images == 0:
            feed_dict_test = {x: x_test_batch, y: y_test_batch, w_loss: loss_weight, lr: learning_rate}
            feed_dict_test_fixed = {x: x_test_fixed, y: y_test_fixed, w_loss: loss_weight, lr: learning_rate}
            y_tested = sess.run(h_pr0, feed_dict=feed_dict_test)
            y_tested_fixed = sess.run(h_pr0, feed_dict=feed_dict_test_fixed)
            new_dir = os.getcwd() + '\\test_images\\' + str(step)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            else:
                print('There already exists a directory: ' + new_dir)
            for i in range(batch_size):
                #[Random test images]Save the results to the folder test_images
                imageio.imwrite(new_dir + '\\TRUEdisp__' + str(step) + '__' + str(i) + '.png', y_test_batch[i][:, :, 0])
                imageio.imwrite(new_dir + '\\PREDdisp__' + str(step) + '__' + str(i) + '.png', y_tested[i][:, :, 0])
                imageio.imwrite(new_dir + '\\img__' + str(step) + '__' + str(i) + '.png', x_test_batch[i][:, :, 0:3])
            for i in range(batch_test_fixed):
                #[Fixed test images]Save the results to the folder test_images
                tmp_dir = os.getcwd() + '\\test_images_fixed\\' + str(i)
                imageio.imwrite(tmp_dir + '\\PREDdisp__' + str(step) + '__' + str(i) + '.png', y_tested_fixed[i][:, :, 0])

            #Randomly pick new test images
            np.random.shuffle(test_file_list)
            x_test_batch, y_test_batch = LoadSamples(batch_size, test_file_list, image_height, image_width, image_downsize)

        if step == 1000:
            loss_weight = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
        if step == 2000:
            loss_weight = np.array([0.0, 0.0, 0.0, 0.33, 0.33, 0.33])
        if step == 3000:
            loss_weight = np.array([0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
        if step == 4000:
            loss_weight = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
        if step == 5000:
            loss_weight = np.array([0.166, 0.166, 0.166, 0.166, 0.166, 0.166])
        if step == 10000:
            loss_weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
        if step == 11000:
            loss_weight = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0])
        if step == 12000:
            loss_weight = np.array([0.33, 0.33, 0.33, 0.0, 0.0, 0.0])
        if step == 13000:
            loss_weight = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        if step == 14000:
            loss_weight = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Save the model when needed
        if step % step_save_model == 0:
            saver.save(sess, os.getcwd() + '\\model_save\\my-cnn-conv-batchN', global_step=step)

