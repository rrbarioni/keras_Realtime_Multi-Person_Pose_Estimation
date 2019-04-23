import math
import os
import re
import sys
import pandas
from functools import partial

import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.model_cmu_mobilenet import get_training_model
from training.optimizers import MultiSGD
from training.dataset_cmu_mobilenet import get_dataflow, batch_dataflow
from training.dataflow import COCODataPaths


batch_size = 10
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

weights_best_file = "weights.best.h5"
training_log = "training.csv"
logs_dir = "./logs"

from_mobilenet = {
}


def get_last_epoch():
    """
    Retrieves last epoch from log file updated during training.

    :return: epoch number
    """
    data = pandas.read_csv(training_log)
    return max(data['epoch'].values)


def restore_weights(weights_best_file, model):
    """
    Restores weights from the checkpoint file if exists or
    preloads the first layers with MobileNet weights

    :param weights_best_file:
    :return: epoch number to use to continue training. last epoch + 1 or 0
    """
    # load previous weights or mobilenet if this is the first run
    if os.path.exists(weights_best_file):
        print("Loading the best weights...")

        model.load_weights(weights_best_file)

        return get_last_epoch() + 1
    else:
        print("Loading mobilenet weights...")

        mobilenet_model = MobileNet(include_top=False, weights='imagenet')
        mobilenet_layer_names_list = [layer.name for layer in mobilenet_model.layers]

        for layer in model.layers:
            if layer.name in mobilenet_layer_names_list:
                mobilenet_layer_name = layer.name
                layer.set_weights(mobilenet_model.get_layer(mobilenet_layer_name).get_weights())
                print("Loaded MobileNet layer: " + mobilenet_layer_name)

        return 0


def get_lr_multipliers(model):
    """
    Setup multipliers for stageN layers (kernel and bias)

    :param model:
    :return: dictionary key: layer name , value: multiplier
    """
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

            # mobilenet
            else:
                kernel_name = layer.weights[0].name
                lr_mult[kernel_name] = 1
                if len(layer.weights) > 1:
                    bias_name = layer.weights[1].name
                    lr_mult[bias_name] = 2

    return lr_mult


def get_loss_funcs():
    """
    Euclidean loss as implemented in caffe
    https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
    :return:
    """
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
    """
    Learning rate schedule - equivalent of caffe lr_policy =  "step"

    :param epoch:
    :param iterations_per_epoch:
    :return:
    """
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

    return lrate


def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i


if __name__ == '__main__':

    # get the model

    model = get_training_model(weight_decay)

    # restore weights

    last_epoch = restore_weights(weights_best_file, model)

    # prepare generators

    curr_dir = os.path.dirname(__file__)
    annot_path_train = os.path.join(curr_dir, '../../COCO-Dataset/annotations/person_keypoints_train2017.json')
    img_dir_train = os.path.abspath(os.path.join(curr_dir, '../../COCO-Dataset/train2017/'))
    annot_path_val = os.path.join(curr_dir, '../../COCO-Dataset/annotations/person_keypoints_val2017.json')
    img_dir_val = os.path.abspath(os.path.join(curr_dir, '../../COCO-Dataset/val2017/'))

    # get dataflow of samples from training set and validation set (we use validation set for training as well)

    coco_data_train = COCODataPaths(
        annot_path=annot_path_train,
        img_dir=img_dir_train
    )
    coco_data_val = COCODataPaths(
        annot_path=annot_path_val,
        img_dir=img_dir_val
    )
    df = get_dataflow([coco_data_train, coco_data_val])
    train_samples = df.size()

    # get generator of batches

    batch_df = batch_dataflow(df, batch_size)
    train_gen = gen(batch_df)

    # setup lr multipliers for conv layers

    lr_multipliers = get_lr_multipliers(model)

    # configure callbacks

    iterations_per_epoch = train_samples // batch_size
    _step_decay = partial(step_decay,
                          iterations_per_epoch=iterations_per_epoch
                          )
    lrate = LearningRateScheduler(_step_decay)
    checkpoint = ModelCheckpoint(weights_best_file, monitor='loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=True, mode='min', period=1)
    csv_logger = CSVLogger(training_log, append=True)
    tb = TensorBoard(log_dir=logs_dir, histogram_freq=0, write_graph=True,
                     write_images=False)

    callbacks_list = [lrate, checkpoint, csv_logger, tb]

    # sgd optimizer with lr multipliers

    multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0,
                        nesterov=False, lr_mult=lr_multipliers)

    # start training

    loss_funcs = get_loss_funcs()
    model.compile(loss=loss_funcs, optimizer=multisgd, metrics=["accuracy"])
    model.fit_generator(train_gen,
                        steps_per_epoch=train_samples // batch_size,
                        epochs=max_iter,
                        callbacks=callbacks_list,
                        use_multiprocessing=False,
                        initial_epoch=last_epoch)
