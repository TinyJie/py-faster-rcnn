# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import _init_paths
import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg

class kitti(imdb):
    def __init__(self, image_set, kitti_path=None):
        imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set
        self._kitti_path = self._get_default_path() if kitti_path is None \
                            else kitti_path
        self._data_path = os.path.join(self._kitti_path, 'data_object_image_2')
        self._classes = ('__background__', 'Pedestrian', 'Car', 'Cyclist')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._kitti_path), \
                'KITTI path does not exist: {}'.format(self._kitti_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/image_2'
        else:
            prefix = 'training/image_2'

        image_path = os.path.join(self._data_path, prefix,
                index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.ROOT_DIR, 'data', 'KITTI')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        """
        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]
        """
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        """

        return gt_roidb

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from file in the kitti format.
        """

        if self._image_set == 'test':
            lines = []
        else:
            filename = os.path.join(self._data_path, 'training', 'label_2', index + '.txt')
            lines = []
            with open(filename) as f:
                for line in f:
                    words = line.split()
                    cls = words[0]
                    truncation = float(words[1])
                    occlusion = int(words[2])
                    height = float(words[7]) - float(words[5])
                    if cls in self._class_to_ind and truncation < 0.5 and occlusion < 3 and height > 25:
                    #if cls in self._class_to_ind:
                        lines.append(line)

        num_objs = len(lines)
        
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for idx, line in enumerate(lines):
            words = line.split()
            cls = self._class_to_ind[words[0]]
            boxes[idx, :] = [float(num) for num in words[4:8]]
            gt_classes[idx] = cls
            overlaps[idx, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes' : gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def evaluate_detections(self, all_boxes, output_dir):
        # for each image
        for im_idx, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
	    print 'Writing KITTI {:s} results to file {:s}'.format(self._image_set, filename)
            with open(filename, 'wt') as f:
                # for each class
                for cls_idx, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_idx][im_idx]
                    if dets == []:
                        continue
		    for k in xrange(dets.shape[0]):
			    f.write('{:s} -1 -1 -10 {:f} {:f} {:f} {:f} -1 -1 -1 -1 -1 -1 -1 {:.32f}\n'.format(\
					cls, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.kitti import kitti
    d = kitti('train')
    res = d.roidb
    from IPython import embed; embed()
