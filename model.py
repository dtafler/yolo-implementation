import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Add, Conv2DTranspose, Conv2D, BatchNormalization, Activation, AveragePooling2D


def conv_block(x, filters, training=True, initializer=keras.initializers.HeNormal()):
    # get filters
    f1, f2 = filters

    # save input for residual
    x_shortcut = x

    x = Conv2D(f1, (1,1), kernel_initializer=initializer)(x)
    x = BatchNormalization()(x, training=training)
    x = Activation('relu')(x)

    x = Conv2D(f2, (3,3), padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x, training=training)
    
    # add residual
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    return x


def build_model(input_shape=(256, 256, 3), classes=0, boxes=2):

    # load pretrained classification model
    base = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False, 
        # with preprocessing, the model expects [0-255] pixel values
        include_preprocessing=True
    )

    # define grid size based on output shape
    base_output_shape = base.layers[-1].output_shape
    grid_len = base_output_shape[1]

    # freeze weights of base model
    base.trainable = False
    for layer in base.layers:
        assert layer.trainable == False
    
    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # double feature map dimensions so not to have too small a grid size
    x = Conv2DTranspose(base_output_shape[-1], (3,3), strides=(2,2), padding='same')(x)

    # pointwise convolution to get desired depth
    x = Conv2D(512, (1,1))(x)

    # residual conv blocks
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2))(x)

    # residual conv blocks
    x = conv_block(x, [512, 1024])
    x = conv_block(x, [512, 1024])

    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1))(x)

    model = keras.Model(inputs, x)
    return model, grid_len


def build_model_no_transpose(input_shape=(256, 256, 3), classes=0, boxes=2):

    # load pretrained classification model
    mobileNet = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        # with preprocessing, the model expects [0-255] pixel values
        include_preprocessing=True
    )

    base = keras.Model(inputs=mobileNet.input, outputs=mobileNet.get_layer('multiply_11').output)

    # freeze weights of base model
    base.trainable = False
    for layer in base.layers:
        assert layer.trainable == False
    
    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # pointwise convolution to get desired depth
    x = Conv2D(512, (1,1))(x)

    # residual conv blocks
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])
    x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2))(x)

    # residual conv blocks
    x = conv_block(x, [512, 1024])
    x = conv_block(x, [512, 1024])

    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1))(x)

    model = keras.Model(inputs, x)

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len


def build_model_no_transpose_large(input_shape=(256, 256, 3), classes=0, boxes=2, blocks_per_section=[4,4]):
    # get number of blocks per section
    b1, b2 = blocks_per_section

    # load pretrained classification model
    mobileNet = keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        # with preprocessing, the model expects [0-255] pixel values
        include_preprocessing=True
    )

    base = keras.Model(inputs=mobileNet.input, outputs=mobileNet.get_layer('multiply_11').output)

    # freeze weights of base model
    base.trainable = False
    for layer in base.layers:
        assert layer.trainable == False
    
    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # pointwise convolution to get desired depth
    x = Conv2D(512, (1,1))(x)

    # residual conv blocks
    for _ in range(b1):
        x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2))(x)

    # residual conv blocks
    for _ in range(b2):
        x = conv_block(x, [512, 1024])

    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1))(x)

    model = keras.Model(inputs, x)

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len

def build_model_effnet(input_shape=(256, 256, 3), classes=0, boxes=2, blocks_per_section=[4,4]):
    # get number of blocks per section
    b1, b2 = blocks_per_section

    # load pretrained classification model
    effnet = keras.applications.EfficientNetV2S(
        include_top=False,
        input_shape=input_shape,
        # with preprocessing, the model expects [0-255] pixel values
        include_preprocessing=True
    )

    base = keras.Model(inputs=effnet.input, outputs=effnet.get_layer('block6a_expand_activation').output)

    # freeze weights of base model
    base.trainable = False
    for layer in base.layers:
        assert layer.trainable == False
    
    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # pointwise convolution to get desired depth
    x = Conv2D(512, (1,1))(x)

    # residual conv blocks
    for _ in range(b1):
        x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2))(x)

    # residual conv blocks
    for _ in range(b2):
        x = conv_block(x, [512, 1024])

    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1))(x)

    model = keras.Model(inputs, x)

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len

class Yolo_head(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        conf = keras.activations.sigmoid(inputs[..., 0:1])
        xy = keras.activations.sigmoid(inputs[..., 1:3])
        wh = tf.math.multiply(keras.activations.sigmoid(inputs[..., 3:5]), 10)
        # the scalar below is the prior for width and height
        # 0.025 * 5 = 0.125, so 5 would be ideal value to predict
        wh = tf.math.multiply(0.025, wh)

        return tf.concat([conf, xy, wh], axis=-1)
    

def build_model_effnet_with_head(input_shape=(256, 256, 3), 
                                 classes=0, boxes=1, blocks_per_section=[4,4]):
    # get number of blocks per section
    b1, b2 = blocks_per_section

    # load pretrained classification model
    effnet = keras.applications.EfficientNetV2S(
        include_top=False,
        input_shape=input_shape,
        # with preprocessing, the model expects [0-255] pixel values
        include_preprocessing=True
    )

    base = keras.Model(inputs=effnet.input, 
                       outputs=effnet.get_layer('block6a_expand_activation').output)

    # freeze weights of base model
    base.trainable = False
    for layer in base.layers:
        assert layer.trainable == False
    
    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # pointwise convolution to get desired depth
    x = Conv2D(512, (1,1))(x)

    # residual conv blocks
    for _ in range(b1):
        x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2), 
               kernel_initializer=keras.initializers.HeNormal())(x)

    # residual conv blocks
    for _ in range(b2):
        x = conv_block(x, [512, 1024])

    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1), 
               kernel_initializer=keras.initializers.HeNormal())(x)
    
    x = BatchNormalization()(x)

    # yolo-head
    x = Yolo_head()(x)

    model = keras.Model(inputs, x, name='yolo')

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len

def build_transfer_model(weights_path, input_shape, classes, boxes, blocks_per_section, num_blocks):
    # build the base model according to parameters
    full_base, _ = build_model_effnet_with_head(input_shape, classes, boxes, blocks_per_section)
    
    # load weights of the model
    full_base.load_weights(weights_path)

    # don't include last conv layer and yolo head
    base = keras.Model(inputs=full_base.input, 
                       outputs=full_base.layers[-4].output)

    # freeze weights
    base.trainable = False

    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # residual conv blocks
    for _ in range(num_blocks):
        x = conv_block(x, [512, 1024])
    
    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1), 
               kernel_initializer=keras.initializers.HeNormal())(x)
    
    x = BatchNormalization()(x)

    # yolo-head
    x = Yolo_head()(x)

    model = keras.Model(inputs, x, name='yolo_transfer')

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len


def build_transfer_model_deep_features(weights_path, input_shape, classes, boxes, blocks_per_section):
    # build the base model according to parameters
    full_base, _ = build_model_effnet_with_head(input_shape, classes, boxes, blocks_per_section)
    
    # load weights of the model
    full_base.load_weights(weights_path)

    # don't include last conv layer and yolo head
    base = keras.Model(inputs=full_base.input, 
                       outputs=full_base.layers[-47].output)

    # freeze weights
    base.trainable = False

    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2), 
               kernel_initializer=keras.initializers.HeNormal())(x)

    # residual conv blocks
    for _ in range(6):
        x = conv_block(x, [512, 1024])
    
    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1), 
               kernel_initializer=keras.initializers.HeNormal())(x)
    
    x = BatchNormalization()(x)

    # yolo-head
    x = Yolo_head()(x)

    model = keras.Model(inputs, x, name='yolo_transfer')

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len


def build_large_transfer_model_deep_features(weights_path, input_shape, classes, boxes, blocks_per_section_base, blocks_per_section):
    # build the base model according to parameters
    full_base, _ = build_model_effnet_with_head(input_shape, classes, boxes, blocks_per_section_base)
    
    # load weights of the model
    full_base.load_weights(weights_path)

    # don't include last conv layer and yolo head
    base = keras.Model(inputs=full_base.input, 
                       outputs=full_base.layers[-47].output)

    # freeze weights
    base.trainable = False

    # define model input
    inputs = Input(shape=input_shape)
    x = base(inputs)

    # residual conv blocks
    for _ in range(blocks_per_section[0]):
        x = conv_block(x, [256, 512])

    # downsampling (fully convolutional instead of pooling)
    x = Conv2D(1024, (3,3), padding='same', strides=(2,2), 
               kernel_initializer=keras.initializers.HeNormal())(x)

    # residual conv blocks
    for _ in range(blocks_per_section[1]):
        x = conv_block(x, [512, 1024])
    
    # pointwise conv for correct output shape
    filters = boxes * 5  + classes
    x = Conv2D(filters, (1,1), 
               kernel_initializer=keras.initializers.HeNormal())(x)
    
    x = BatchNormalization()(x)

    # yolo-head
    x = Yolo_head()(x)

    model = keras.Model(inputs, x, name='yolo_transfer')

    grid_len = model.layers[-1].output_shape[1]
    return model, grid_len