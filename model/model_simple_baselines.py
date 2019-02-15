from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers import BatchNormalization, add
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant


# mask1 = AveragePooling2D((8, 8), strides=(8, 8))(mask1)
# mask2 = AveragePooling2D((8, 8), strides=(8, 8))(mask2)
def apply_mask(x, mask1, mask2, num_p, branch):
    w_name = "weight_L%d" % branch
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight
    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def resnet50_block(x, weight_decay):
    def conv_block(x, nf, ks, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = nf
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut_1 = Conv2D(filters1, (1, 1), strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '2a')(x)
        shortcut_1 = BatchNormalization(axis=3,
            name=bn_name_base + '2a')(shortcut_1)
        shortcut_1 = Activation('relu')(shortcut_1)

        shortcut_1 = Conv2D(filters2, ks, padding='same',
            kernel_initializer='he_normal',
            name=conv_name_base + '2b')(shortcut_1)
        shortcut_1 = BatchNormalization(axis=3,
            name=bn_name_base + '2b')(shortcut_1)
        shortcut_1 = Activation('relu')(shortcut_1)

        shortcut_1 = Conv2D(filters3, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(shortcut_1)
        shortcut_1 = BatchNormalization(axis=3,
            name=bn_name_base + '2c')(shortcut_1)

        shortcut_2 = Conv2D(filters3, (1, 1), strides=strides,
            kernel_initializer='he_normal',
            name=conv_name_base + '1')(x)
        shortcut_2 = BatchNormalization(axis=3,
            name=bn_name_base + '1')(shortcut_2)

        x = add([shortcut_1, shortcut_2])
        x = Activation('relu')(x)
        return x

    def identity_block(x, nf, ks, stage, block):
        filters1, filters2, filters3 = nf
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        shortcut = Conv2D(filters1, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2a')(x)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '2a')(shortcut)
        shortcut = Activation('relu')(shortcut)

        shortcut = Conv2D(filters2, ks,
            padding='same', kernel_initializer='he_normal',
            name=conv_name_base + '2b')(shortcut)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '2b')(shortcut)
        shortcut = Activation('relu')(shortcut)

        shortcut = Conv2D(filters3, (1, 1),
            kernel_initializer='he_normal',
            name=conv_name_base + '2c')(shortcut)
        shortcut = BatchNormalization(axis=3,
            name=bn_name_base + '2c')(shortcut)

        x = add([shortcut, x])
        x = Activation('relu')(x)

        return x

    #Block 1
    x = ZeroPadding2D((3, 3), name='conv1_pad')(x)
    x = Conv2D(64, (7, 7), strides=(2, 2),
        padding='valid', kernel_initializer='he_normal',
        name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    #Block 2
    x = conv_block(x, [64, 64, 256], 3, stage=2, block='a',
        strides=(1, 1))
    x = identity_block(x, [64, 64, 256], 3, stage=2, block='b')
    x = identity_block(x, [64, 64, 256], 3, stage=2, block='c')

    #Block 3
    x = conv_block(x, [128, 128, 512], 3, stage=3, block='a')
    x = identity_block(x, [128, 128, 512], 3, stage=3, block='b')
    x = identity_block(x, [128, 128, 512], 3, stage=3, block='c')
    x = identity_block(x, [128, 128, 512], 3, stage=3, block='d')

    #Block 4
    x = conv_block(x, [256, 256, 1024], 3, stage=4, block='a')
    x = identity_block(x, [256, 256, 1024], 3, stage=4, block='b')
    x = identity_block(x, [256, 256, 1024], 3, stage=4, block='c')
    x = identity_block(x, [256, 256, 1024], 3, stage=4, block='d')
    x = identity_block(x, [256, 256, 1024], 3, stage=4, block='e')
    x = identity_block(x, [256, 256, 1024], 3, stage=4, block='f')

    #Block 5
    x = conv_block(x, [512, 512, 2048], 3, stage=5, block='a')
    x = identity_block(x, [512, 512, 2048], 3, stage=5, block='b')
    x = identity_block(x, [512, 512, 2048], 3, stage=5, block='c')

    return x


def deconv_block(x, nf, ks, strides, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None
    
    x = Conv2DTranspose(nf, (ks, ks),
        strides=strides,
        padding='same',
        activation='relu',
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        kernel_initializer=random_normal(stddev=0.01),
        bias_initializer=constant(0.0))(x)

    return x


def lastconv_block(x, nf, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (1, 1),
        padding='same',
        kernel_regularizer=kernel_reg,
        bias_regularizer=bias_reg,
        kernel_initializer=random_normal(stddev=0.01),
        bias_initializer=constant(0.0))(x)

    return x
    

def get_training_model(weight_decay):
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)
    # img_input_shape = (368, 368, 3)
    # vec_input_shape = (368, 368, 38)
    # heat_input_shape = (368, 368, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    resnet50_out = resnet50_block(img_normalized, (weight_decay, 0))

    x = resnet50_out
    x = deconv_block(x, 256, 4, (2, 2), (weight_decay, 0))
    x = deconv_block(x, 256, 4, (2, 2), (weight_decay, 0))
    x = Lambda(lambda x: x[:,1:-1,1:-1,:])(x)
    # x = deconv_block(x, 256, 4, (2, 2), (weight_decay, 0))

    branch1_out = lastconv_block(x, np_branch1, (weight_decay, 0))
    w1 = apply_mask(branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1)

    branch2_out = lastconv_block(x, np_branch2, (weight_decay, 0))
    w2 = apply_mask(branch2_out, vec_weight_input, heat_weight_input, np_branch2, 2)

    outputs.append(w1)
    outputs.append(w2)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    np_branch1 = 38
    np_branch2 = 19

    # img_input_shape = (None, None, 3)
    img_input_shape = (368, 368, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    resnet50_out = resnet50_block(img_normalized, None)

    x = resnet50_out
    x = deconv_block(x, 256, 4, (2, 2), None)
    x = deconv_block(x, 256, 4, (2, 2), None)
    x = Lambda(lambda x: x[:,1:-1,1:-1,:])(x)
    # x = deconv_block(x, 256, 4, (2, 2), None)

    branch1_out = lastconv_block(x, np_branch1, None)
    branch2_out = lastconv_block(x, np_branch2, None)

    model = Model(inputs=[img_input], outputs=[branch1_out, branch2_out])

    return model
