from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ReLU, BatchNormalization, \
    DepthwiseConv2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant


def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, st, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), strides=(st, st), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)

    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    
    return x


def mobilenet_block(x, weight_decay):
    def conv_block(inputs, filters, kernel, st, weight_decay):
        channel_axis = -1
        x = ZeroPadding2D(padding=((0, 1), (0, 1)),
            name='conv1_pad')(inputs)
        x = Conv2D(filters, (kernel, kernel), padding='valid',
            use_bias=False, strides=(st, st), name='conv1')(x)
        # x = conv(x, filters, kernel, st, 'conv1', (weight_decay, 0))
        x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        x = ReLU(6., name='conv1_relu')(x)

        return x

    def depthwise_conv_block(inputs, pointwise_conv_filters, st,
        block_id, weight_decay):
        channel_axis = -1
        if st == 1:
            x = inputs
        else:
            x = ZeroPadding2D(((0, 1), (0, 1)),
                name='conv_pad_%d' % block_id)(inputs)
        x = DepthwiseConv2D((3, 3),
            padding='same' if st == 1 else 'valid',
            depth_multiplier=1, strides=(st, st), use_bias=False,
            name='conv_dw_%d' % block_id)(x)
        x = BatchNormalization(axis=channel_axis,
            name='conv_dw_%d_bn' % block_id)(x)
        x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)


        x = Conv2D(pointwise_conv_filters, (1, 1), padding='same',
            use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
        # x = conv(x, pointwise_conv_filters, 1, 1, 'conv_pw_%d' % block_id,
        #     (weight_decay, 0))
        x = BatchNormalization(axis=channel_axis,
            name='conv_pw_%d_bn' % block_id)(x)
        x = ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

        return x

    x = conv_block(x, 32, 3, 2, weight_decay)
    x = depthwise_conv_block(x, 64, 1, 1, weight_decay)

    x = depthwise_conv_block(x, 128, 2, 2, weight_decay)
    x = depthwise_conv_block(x, 128, 1, 3, weight_decay)

    x = depthwise_conv_block(x, 256, 2, 4, weight_decay)
    x = depthwise_conv_block(x, 256, 1, 5, weight_decay)

    x = depthwise_conv_block(x, 512, 2, 6, weight_decay)
    x = depthwise_conv_block(x, 512, 1, 7, weight_decay)
    x = depthwise_conv_block(x, 512, 1, 8, weight_decay)
    x = depthwise_conv_block(x, 512, 1, 9, weight_decay)
    x = depthwise_conv_block(x, 512, 1, 10, weight_decay)
    x = depthwise_conv_block(x, 512, 1, 11, weight_decay)

    x = depthwise_conv_block(x, 1024, 2, 12, weight_decay)
    x = depthwise_conv_block(x, 1024, 1, 13, weight_decay)

    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, 1, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, 1, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, 1, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, 1, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, 1, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, 1, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, 1, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, 1, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x


# mask1 = AveragePooling2D((8, 8), strides=(8, 8))(mask1)
# mask2 = AveragePooling2D((8, 8), strides=(8, 8))(mask2)
def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight
    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)
    # img_input_shape = (224, 224, 3)
    # vec_input_shape = (224, 224, 38)
    # heat_input_shape = (224, 224, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # MobileNet
    stage0_out = mobilenet_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages = 6
    np_branch1 = 38
    np_branch2 = 19

    # img_input_shape = (None, None, 3)
    img_input_shape = (224, 224, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # MobileNet
    stage0_out = mobilenet_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model
