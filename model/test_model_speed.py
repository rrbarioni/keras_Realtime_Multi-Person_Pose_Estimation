import numpy as np
import time

eval_cmu = False
eval_simple_baselines = not eval_cmu

eval_testing_model = True
eval_training_model = not eval_testing_model

input_size = 368
num_keypoints = 19
num_pafs = 38

if eval_testing_model:
    if eval_cmu:
        from model_cmu import get_testing_model as get_model
    else:
        from model_simple_baselines import get_testing_model as get_model
    model = get_model()
else:
    if eval_cmu:
        from model_cmu import get_training_model as get_model
    else:
        from model_simple_baselines import get_training_model as get_model
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
