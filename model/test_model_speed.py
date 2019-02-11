import numpy as np
import time

# from model_cmu import get_testing_model
# from model_simple_baselines import get_testing_model
from model_simple_baselines import get_training_model

# model = get_testing_model()
model = get_training_model(5e-4)
model.summary()

zeros = np.zeros((1, 368, 368, 3))
vec_zeros = np.zeros((1, 368, 368, 38))
heat_zeros = np.zeros((1, 368, 368, 19))
times = np.array([])
# model.predict(zeros)
model.predict([zeros, vec_zeros, heat_zeros])

for i in range(10):
   t0 = time.time()
   model.predict(zeros)
   curr_time = time.time() - t0
   times = np.append(times, curr_time)
   print(curr_time)

print('mean: %s' % np.mean(times))
