import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import util
import scipy.ndimage.filters as fi

from model.model_cmu_egocap import get_testing_model

def predict_joints(model, input_image):
    oriImg = cv2.imread(input_image)
    
    scale = model_params['boxsize'] / oriImg.shape[0]

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
        model_params['stride'], model_params['padValue'])

    input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
        (3,0,1,2)) # required shape (1, width, height, channels)

    output_blobs = model.predict(input_img)

    # extract outputs, resize, and remove padding
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'],
        fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[
        :imageToTest_padded.shape[0] - pad[2],
        :imageToTest_padded.shape[1] - pad[3],
        :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]),
        interpolation=cv2.INTER_CUBIC)

    all_peaks = [np.unravel_index(heatmap[:,:,i].argmax(), heatmap[:,:,i].shape)    
        for i in range(heatmap.shape[-1])]

    return all_peaks

def create_canvas(input_image, all_peaks):
    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(len(all_peaks) - 1):
        cv2.circle(canvas, all_peaks[i][::-1], 4, colors[i], thickness=-1)
    
    for (i, j) in joint_index_pairs:
        i_x, i_y = all_peaks[i]
        j_x, j_y = all_peaks[j]
        m_y = np.mean([i_y, j_y])
        m_x = np.mean([i_x, j_x])
    
        length = ((i_x - j_x) ** 2 + (i_y - j_y) ** 2) ** 0.5
        angle = math.degrees(math.atan2(i_x - j_x, i_y - j_y))
    
        curr_canvas = canvas.copy()
        polygon = cv2.ellipse2Poly((int(m_y), int(m_x)),
            (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(curr_canvas, polygon, colors[j-1])
        canvas = cv2.addWeighted(canvas, 0.4, curr_canvas, 0.6, 0)
        
    return canvas
    

params = { 'scale_search': [1], 'thre1': 0.1, 'thre2': 0.05, 'mid_num': 10 }

model_params = { 'boxsize': 368, 'stride': 8, 'padValue': 128 }

joint_index_pairs = list(zip(
    [0, 1, 2, 3, 4, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))

# visualize
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]

input_image_1 = 'sample_images/egocap/S5_v002_cam1_frame-2324.jpg'
input_image_2 = 'sample_images/egocap/S5_v002_cam1_frame-2326.jpg'
keras_weights_file = 'training/results/cmu_egocap/weights.h5'

model = get_testing_model()
model.load_weights(keras_weights_file)

all_peaks_1 = predict_joints(model, input_image_1)
all_peaks_2 = predict_joints(model, input_image_2)

canvas_1 = create_canvas(input_image_1, all_peaks_1)
canvas_2 = create_canvas(input_image_2, all_peaks_2)

plt.imshow(canvas_1)
plt.imshow(canvas_2)
