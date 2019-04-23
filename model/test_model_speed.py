import numpy as np
import os
import time
# import cProfile

eval_testing_model = True
cpu = False
model_arch = 'cmu_mobilenet'

input_size = 224
num_keypoints = 19
num_pafs = 38

if cpu:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if eval_testing_model:
    exec('from model_%s import get_testing_model as get_model' % model_arch)
    model = get_model()
else:
    exec('from model_%s import get_training_model as get_model' % model_arch)
    model = get_model(5e-4)

model.summary()

if eval_testing_model:
    zeros = np.zeros((1, input_size, input_size, 3))
    
    model.predict(zeros)
else:
    zeros = np.zeros((1, input_size, input_size, 3))
    vec_zeros = np.zeros((1, input_size, input_size, num_pafs))
    heat_zeros = np.zeros((1, input_size, input_size, num_keypoints))

    model.predict([zeros, vec_zeros, heat_zeros])

'''
if eval_testing_model:
    cProfile.run('model.predict(zeros)')
else:
    cProfile.run('model.predict([zeros, vec_zeros, heat_zeros])')
'''


times = np.array([])
for i in range(10):
    t0 = time.time()
    
    if eval_testing_model:
        model.predict(zeros)
    else:
        model.predict([zeros, vec_zeros, heat_zeros])

    curr_time = time.time() - t0
    times = np.append(times, curr_time)
    print(curr_time)

print('mean: %s' % np.mean(times))

