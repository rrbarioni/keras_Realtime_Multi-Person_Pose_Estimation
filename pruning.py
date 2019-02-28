import numpy as np
from keras.models import load_model, Model

from model.model_simple_baselines import get_testing_model as get_model

model = get_model()
model.load_weights('training/results/simple_baselines/weights.h5')

model.summary()

def get_l1_list_per_layer(model):
    l1_list_per_layer = []
    last_i = 30
    for i in range(len(model.layers[:last_i])):
        l = model.layers[i]
        lw = l.get_weights()
        
        if len(lw) == 2:
            lwnp = np.array(lw)
            
            l1_list_per_layer.append(
                np.array(
                    [sum(sum(abs(lwnp[0][...,i,j])))
                    for i in range(lwnp[0].shape[-2])
                    for j in range(lwnp[0].shape[-1])]
                )
            )
            
        else:
            l1_list_per_layer.append(np.array([]))
            
        print('Done %s out of %s layers' % (i, len(model.layers[:last_i])))
        
    l1_list_per_layer = np.array(l1_list_per_layer)
        
    return l1_list_per_layer

l1_list_per_layer = get_l1_list_per_layer(model)
