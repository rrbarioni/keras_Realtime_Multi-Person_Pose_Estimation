import os
import numpy as np
import random

from tensorpack.dataflow.base import RNGDataFlow

class JointsLoader:
    '''
    Loader for joints from egocap keypoints
    '''
    num_joints = 18

    num_joints_and_bkg = num_joints + 1

    num_connections = 17

    '''
    head*, neck, left shoulder, left elbow, left wrist, left finger,
    right shoulder, right elbow, right wrist, right finger, left hip, left knee,
    left ankle, left toe, right hip, right knee, right ankle, right toe
    '''

    idx_in_egocap = [0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 14, 15,
        16, 17, 10, 11, 12, 13]

    idx_in_egocap_str = ['Head', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
        'RFinger', 'LShoulder', 'LElbow', 'LWrist', 'LFinger', 'RHip', 'RKnee',
        'RAnkle', 'RToe', 'LHip', 'LKnee', 'LAnkle', 'RToe']

    joint_pairs = list(zip(
        [0, 1, 2, 3, 4, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))

class Meta(object):
    '''
    Metadata representing a single data point for training.
    '''
    __slots__ = (
        'img_path',
        'height',
        'width',
        'center',
        'num_keypoints',
        'masks_segments',
        'all_joints',
        'img',
        'mask',
        'aug_center',
        'aug_joints',
        'horizontal_flip')

    def __init__(self, img_path, height, width, num_keypoints, all_joints):
        self.img_path = img_path
        self.height = height
        self.width = width
        self.center = np.expand_dims([int(self.height / 2), int(self.width / 2)], axis=0)
        self.num_keypoints = num_keypoints

        # updated after iterating over all persons
        self.masks_segments = []
        self.all_joints = all_joints

        # updated during augmentation
        self.img = None
        self.mask = None
        self.aug_center = None
        self.aug_joints = None

        # as there are two cameras (left and right)
        self.horizontal_flip = bool(random.randint(0, 1))

class EgoCapDataPaths:
    """
    Holder for egocap dataset paths
    """
    def __init__(self, annot_path, img_dir):
        self.img_dir = img_dir

        self.annot = []

        i = 0
        with open(annot_path, 'r') as f:
            while len(f.readline()) != 0:
                curr_img_path = f.readline()
                curr_img_path = curr_img_path[
                    curr_img_path.rfind('/') + 1:curr_img_path.find('\\')]

                f.readline()

                curr_height = f.readline()
                curr_height = int(curr_height.replace('\n', ''))

                curr_width = f.readline()
                curr_width = int(curr_width.replace('\n', ''))

                curr_num_keypoints = f.readline()
                curr_num_keypoints = int(curr_num_keypoints.replace('\n', ''))

                curr_keypoints = []
                for k in range(curr_num_keypoints):
                    curr_k = f.readline()
                    curr_k = curr_k.replace('\n', '')
                    _, curr_x, curr_y = curr_k.split(' ')
                    curr_x = int(curr_x)
                    curr_y = int(curr_y)

                    if curr_x >=0 and curr_y >= 0 and curr_x < curr_width \
                        and curr_y < curr_height:
                        curr_keypoints.append((curr_x, curr_y))
                    else:
                        curr_keypoints.append(None)

                self.annot.append({
                    'img_path': curr_img_path,
                    'height': curr_height,
                    'width': curr_width,
                    'num_keypoints': curr_num_keypoints,
                    'keypoints': curr_keypoints
                })

                i += 1
                if i % 10000 == 0:
                    print("Loading data paths {}".format(i))


class EgoCapDataFlow(RNGDataFlow):
    '''
    Tensorpack dataflow serving coco data points.
    '''
    def __init__(self, target_size, egocap_data, select_ids=None):
        '''
        Initializes dataflow.

        :param target_size:
        :param egocap_data: paths to the egocap files: annotation file and
            folder with images
        :param select_ids: (optional) identifiers of images to serve
            (for debugging)
        '''
        self.egocap_data = egocap_data if isinstance(egocap_data, list) \
            else [egocap_data]
        self.all_meta = []
        self.select_ids = select_ids
        self.target_size = target_size

    def __iter__(self):
        return get_data(self)

    def __len__(self):
        return len(self.all_meta)

    def prepare(self):
        '''
        Loads egocap metadata.
         ['all_joints', 'aug_center', 'aug_joints',
         'height', 'img', 'img_path', 'mask', 'masks_segments', 'num_keypoints',
         'scale', 'width']
        '''
        for egocap in self.egocap_data:

            print("Loading dataset {} ...".format(egocap.img_dir))

            for i, curr_annot in enumerate(egocap.annot):
                self.all_meta.append(Meta(
                    img_path=os.path.join(
                        egocap.img_dir, curr_annot['img_path']),
                    height=curr_annot['height'],
                    width=curr_annot['width'],
                    num_keypoints=curr_annot['num_keypoints'],
                    all_joints=[curr_annot['keypoints']]))

                if i % 10000 == 0:
                    print("Loading image annot {}/{}".format(i, len(egocap.annot)))

def save(self, path):
    raise NotImplemented

def load(self, path):
    raise NotImplemented

def size(self):
    '''
    :return: number of items
    '''
    return len(self.all_meta)

def get_data(self):
    '''
    Generator of data points

    :return: instance of Meta
    '''
    # idxs = np.arange(self.size())
    idxs = np.arange(len(self))
    self.rng.shuffle(idxs)
    for idx in idxs:
        yield [self.all_meta[idx]]
