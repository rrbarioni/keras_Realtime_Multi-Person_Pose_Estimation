import numpy as np

import keras
from keras.models import load_model, Model
from kerassurgeon import Surgeon

from model.model_cmu import get_testing_model as get_model

np.set_printoptions(suppress=True)

model = get_model()
model.load_weights('training/results/cmu/weights.h5')

model.summary()

'''
l1_list_per_layer:
    list of quartets ('a', 'b', 'c'), where:
        a: index of model layer
        b: index of filter at layer 'a'
        c: l1 norm of filter ('a', 'b')
'''
def get_l1_list_per_layer(model):
    l1_list_per_layer = []
    last_i = len(model.layers)
    for i in range(last_i):
        l = model.layers[i]
        lwnp = np.array(l.get_weights())
        
        if len(lwnp) == 2:
            filter_length = lwnp[0].shape[-2]
            if type(l) == keras.layers.convolutional.Conv2DTranspose:
                bias_length = lwnp[0].shape[-2]
            else:
                bias_length = lwnp[0].shape[-1]
            
            lwnp_l1_list = np.array(
                [(i, (j * lwnp[0].shape[-1]) + k,
                  sum(sum(abs(lwnp[0][...,j,k]))) + abs(lwnp[1][k]))
                for j in range(filter_length)
                for k in range(bias_length)]
            )
            
            for l1 in lwnp_l1_list:
                l1_list_per_layer.append(l1)
            
        print('Done %s out of %s layers' % (i, len(model.layers[:last_i])))
        
    l1_list_per_layer = np.array(l1_list_per_layer)
        
    return l1_list_per_layer

l1_list_per_layer = get_l1_list_per_layer(model)
sorted_l1_list_per_layer = l1_list_per_layer[l1_list_per_layer[:,2].argsort()]

prune_l1_5percent_list = sorted_l1_list_per_layer[
    :int(len(sorted_l1_list_per_layer) * 0.05)]
prune_l1_5percent_by_layer = {
    int(l): list(prune_l1_5percent_list[prune_l1_5percent_list[:,0] == l][:,1]
        .astype(int))
    for l in np.unique(prune_l1_5percent_list[:,0]) }

surgeon = Surgeon(model)
for (l, fl) in prune_l1_5percent_by_layer.items():
    surgeon.add_job('delete_channels', model.layers[l],
        channels=fl)
surgeon.operate()
