from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers import BatchNormalization, add
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant


def relu(x): return Activation('relu')(x)


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def resnet50_block(x, weight_decay):
    def conv_block(x, ks, nf, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = nf
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '2a')(x)
        x = BatchNormalization(axis=3,
            name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, ks, padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3,
            name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3,
            name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '1')(x)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(x, ks, nf, stage, block):
        filters1, filters2, filters3 = nf
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = Conv2D(filters1, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2a')(x)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '2a')(shortcut)
        shortcut = Activation('relu')(shortcut)

        shortcut = Conv2D(filters2, kernel_size,
            padding='same', kernel_initializer='he_normal',
            name=conv_name_base + '2b')(shortcut)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '2b')(shortcut)
        shortcut = Activation('relu')(shortcut)

        shortcut = Conv2D(filters3, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(shortcut)
        shortcut = BatchNormalization(axis=bn_axis,
            name=bn_name_base + '2c')(shortcut)

        x = add([shortcut, x])
        x = Activation('relu')(x)

        return x

    #Block 1
    x = ZeroPadding2D((3, 3), name='conv1_pad')
    x = Conv2D(64, (7, 7), strides=(2, 2),
        padding='valid', kernel_initializer='he_normal',
        name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #Block 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a',
        strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    #Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    #Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    #Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x


def get_training_model(weight_decay):
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    resnet50_out = resnet50_block(img_normalized, weight_decay)

    return model


def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)

    return model