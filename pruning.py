import numpy as np
import keras
from keras.models import load_model, Model

from model.model_simple_baselines import get_testing_model as get_model

np.set_printoptions(suppress=True)

model = get_model()
model.load_weights('training/results/simple_baselines/weights.h5')

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
