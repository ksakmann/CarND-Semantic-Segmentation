import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    g = sess.graph
     
    # print(sess.graph_def)   
    image_input = g.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = g.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = g.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = g.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = g.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

#def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
#    return tf.truncated_normal(shape, dtype=dtype, stddev = 0.01, seed=seed)

#def conv_1x1(x, num_outputs, activation = tf.nn.relu):
def conv_1x1(x, num_outputs, activation = None):
    """
    Perform a 1x1 convolution 
    :x: 4-Rank Tensor
    :return: TF Operation
    """
    kernel_size = 1
    stride = 1

    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.layers.conv2d(x, num_outputs, 
        kernel_size = (1,1),
        strides = (1,1),
        padding = 'SAME',
        activation = activation,  # TODO: test different activations
        kernel_initializer = initializer)

def upsample(x,filters,kernel_size,strides):    
    """
    Apply a two times upsample on x and return the result.
    :x: 4-Rank Tensor
    :return: TF Operation
    """
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.layers.conv2d_transpose(x,filters,kernel_size, strides,padding = 'SAME', kernel_initializer = initializer)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    layer7_1x1 = conv_1x1(vgg_layer7_out,num_classes)
    layer4_1x1 = conv_1x1(vgg_layer4_out,num_classes)
    layer3_1x1 = conv_1x1(vgg_layer3_out,num_classes)

    layer7_up = upsample(layer7_1x1,num_classes,5,2) 
    layer4_skip = tf.add(layer7_up,layer4_1x1)

    layer4_up = upsample(layer4_skip,num_classes,5,2)
    layer3_skip = tf.add(layer4_up,layer3_1x1)

    layer3_up = upsample(layer3_skip, num_classes,16,8)

    return layer3_up


#    init = tf.truncated_normal_initializer(stddev = 0.01)
#    def conv_1x1(x, num_classes, init = init):
#        return tf.layers.conv2d(x, num_classes, 1, padding = 'same', kernel_initializer = init)
#
#    def upsample(x, num_classes, depth, strides, init = init):
#        return tf.layers.conv2d_transpose(x, num_classes, depth, strides, padding = 'same', kernel_initializer = init)
#
#    layer_7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
#    layer_4_1x1 = conv_1x1(vgg_layer4_out, num_classes)
#    layer_3_1x1 = conv_1x1(vgg_layer3_out, num_classes)
#
#    upsample1 = upsample(layer_7_1x1, num_classes, 5, 2)
#    layer1 = tf.layers.batch_normalization(upsample1)
#    layer1 = tf.add(layer1, layer_4_1x1)
#
#
#    upsample2 = upsample(layer1, num_classes, 5, 2)
#    layer2 = tf.layers.batch_normalization(upsample2)
#    layer2 = tf.add(layer2, layer_3_1x1)
#
#
#    return upsample(layer2, num_classes, 14, 8)



tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, [-1,num_classes])
    labels = tf.reshape(correct_label,[-1,num_classes])
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')
    train_op = optimizer.minimize(loss=cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function


    for epoch_i in range(epochs):
        
    # Progress bar
    #batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        counter = 1
        for images, correct_labels in get_batches_fn(batch_size):

            # Run optimizer and get loss
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: images, correct_label: correct_labels, keep_prob: 0.8, learning_rate: 0.001}
                )

            if counter % 10 == 0: 
                print("loss", loss, " epoch_i ", epoch_i, " epochs ", epochs)
            counter += 1                

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    
    epochs = 12
    batch_size = 13

    data_dir = './data'
    runs_dir = './runs'
    #model_path ='./tmp/fcn.ckpt'
    #model_path ='./tmp/fcn_e20_b13.ckpt'
    model_path ='./tmp/fcn_e20_b13_linear.ckpt'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)


    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        learning_rate = tf.placeholder(dtype = tf.float32)
        correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, num_classes)) 

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

#        print("nn_last_layer.get_shape()",nn_last_layer.get_shape())

#        out = sess.run(nn_last_layer,feed_dict = {input_image: images,keep_prob: 0.5})


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
