import math
import os
import re
import sys
import numpy as np
import pandas as pd
from functools import partial

import tensorflow as tf

import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, \
    TensorBoard
from keras.layers.convolutional import Conv2D
from keras.models import load_model

from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.model_cmu import get_training_model
from training.optimizers import MultiSGD
from training.dataset_cmu import get_dataflow, batch_dataflow
from training.dataflow import COCODataPaths


batch_size = 10
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053 
# after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

percent_pruning = 2
total_percent_pruning = 50

model_type = 'pruned_cmu'
output_dir = os.path.join('results', model_type)
# weights_file = os.path.join(output_dir, 'weights.h5')
training_log = 'training.csv'
logs_dir = './logs'

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():
    '''
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    '''
    data = pd.read_csv(training_log)
    return max(data['epoch'].values)


def restore_weights(weights_file, model):
    '''
    Restores weights from the checkpoint file if exists or
    preloads the first layers with VGG19 weights

    :param weights_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    '''
    # load previous weights or vgg19 if this is the first run
    if os.path.exists(weights_file):
        print("Loading the best weights...")

        model.load_weights(weights_file)

        return get_last_epoch() + 1
    else:
        print("Loading vgg19 weights...")

        vgg_model = VGG19(include_top=False, weights='imagenet')

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(
                    vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        return 0


def get_lr_multipliers(model):
    '''
    Setup multipliers for stageN layers (kernel and bias)

    :param model:
    :return: dictionary key: layer name , value: multiplier
    '''
    lr_mult = dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

    return lr_mult


def get_loss_funcs():
    '''
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    '''
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2

    losses = {}
    losses["weight_stage1_L1"] = _eucl_loss
    losses["weight_stage1_L2"] = _eucl_loss
    losses["weight_stage2_L1"] = _eucl_loss
    losses["weight_stage2_L2"] = _eucl_loss
    losses["weight_stage3_L1"] = _eucl_loss
    losses["weight_stage3_L2"] = _eucl_loss
    losses["weight_stage4_L1"] = _eucl_loss
    losses["weight_stage4_L2"] = _eucl_loss
    losses["weight_stage5_L1"] = _eucl_loss
    losses["weight_stage5_L2"] = _eucl_loss
    losses["weight_stage6_L1"] = _eucl_loss
    losses["weight_stage6_L2"] = _eucl_loss

    return losses


def step_decay(epoch, iterations_per_epoch):
    '''
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    '''
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate


def gen(df):
    '''
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    '''
    while True:
        for i in df.get_data():
            yield i


def prune_model(model, apoz_df, n_channels_delete):
    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    # Get APoZ
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df


if __name__ == '__main__':

    # get the model
    model = get_training_model(weight_decay)

    # restore weights
    # last_epoch = restore_weights(weights_file, model)

    # prepare generators
    curr_dir = os.path.dirname(__file__)
    annot_path_train = os.path.join(curr_dir,
        '../../COCO-Dataset/annotations/person_keypoints_train2017.json')
    img_dir_train = os.path.abspath(os.path.join(curr_dir,
        '../../COCO-Dataset/train2017/'))
    annot_path_val = os.path.join(curr_dir,
        '../../COCO-Dataset/annotations/person_keypoints_val2017.json')
    img_dir_val = os.path.abspath(os.path.join(curr_dir,
        '../../COCO-Dataset/val2017/'))

    '''
    get dataflow of samples from training set and validation set
        (we use validation set for training as well)
    '''
    coco_data_train = COCODataPaths(
        annot_path=annot_path_train,
        img_dir=img_dir_train
    )
    coco_data_val = COCODataPaths(
        annot_path=annot_path_val,
        img_dir=img_dir_val
    )
    train_df = get_dataflow([coco_data_train])
    train_samples = train_df.size()
    val_df = get_dataflow([coco_data_val])

    # get generator of batches
    train_batch_df = batch_dataflow(train_df, batch_size)
    train_gen = gen(train_batch_df)
    val_batch_df = batch_dataflow(val_df, batch_size)
    val_gen = gen(val_batch_df)

    # setup lr multipliers for conv layers
    lr_multipliers = get_lr_multipliers(model)

    # configure callbacks
    iterations_per_epoch = train_samples // batch_size
    _step_decay = partial(step_decay, iterations_per_epoch=iterations_per_epoch)
    lrate = LearningRateScheduler(_step_decay)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
        write_images=False)

    # sgd optimizer with lr multipliers
    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
                        nesterov=False, lr_mult=lr_multipliers)

    # start training
    loss_funcs = get_loss_funcs()
    model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])

    total_channels = get_total_channels(model)
    n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))

    # Incrementally prune the network, retraining it each time
    percent_pruned = 0
    # If percent_pruned > 0, continue pruning from previous checkpoint
    if percent_pruned > 0:
        checkpoint_name = ('%s_%s_percent' % (model_type, percent_pruned))
        model.load_weights(output_dir + checkpoint_name + '.h5')

    while percent_pruned <= total_percent_pruning:
        # Prune the model
        apoz_df = get_model_apoz(model, val_gen)
        percent_pruned += percent_pruning
        print('pruning up to %s percent of the original model weights' %
            percent_pruned)
        model = prune_model(model, apoz_df, n_channels_delete)

        # Clean up tensorflow session after pruning and re-load model
        checkpoint_name = ('%s_%s_percent' % (model_type, percent_pruned))
        model.save(output_dir + checkpoint_name + '.h5')
        del model
        K.clear_session()
        tf.reset_default_graph()
        model = get_training_model(weight_decay)
        model.load_weights(output_dir + checkpoint_name + '.h5')

        model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])
        checkpoint = ModelCheckpoint(weights_best_file, monitor='loss',
            verbose=0, save_best_only=False, save_weights_only=True, mode='min',
            period=1)
        callbacks_list = [lrate, checkpoint, csv_logger, tb]

        model.fit_generator(train_gen,
                            steps_per_epoch=train_samples // batch_size,
                            epochs=max_iter,
                            callbacks=callbacks_list,
                            use_multiprocessing=False,
                            initial_epoch=last_epoch)
