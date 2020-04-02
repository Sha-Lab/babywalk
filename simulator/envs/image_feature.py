import numpy as np
import sys
import paths
import csv
from collections import defaultdict
import torch
from torch.autograd import Variable

from envs_utils import decode_base64
from model.cuda import try_cuda

csv.field_size_limit(sys.maxsize)

class ImageFeatures(object):
    NUM_VIEWS = 36
    MEAN_POOLED_DIM = 2048
    feature_dim = MEAN_POOLED_DIM
    shape = (NUM_VIEWS, feature_dim)

    IMAGE_W = 640
    IMAGE_H = 480
    VFOV = 60

    @staticmethod
    def from_args(args):
        for image_feature_type in sorted(args.image_feature_type):
            assert image_feature_type == "mean_pooled"
        return [MeanPooledImageFeatures(args.image_feature_datasets)]

    @staticmethod
    def add_args(argument_parser):
        argument_parser.add_argument("--image_feature_type", nargs="+",
                                     choices=["none", "mean_pooled", "convolutional_attention", "bottom_up_attention"],
                                     default=["mean_pooled"])
        argument_parser.add_argument("--image_feature_datasets", nargs="+", choices=["imagenet", "places365"],
                                     default=["imagenet"],
                                     help="only applicable to mean_pooled or convolutional_attention options for --image_feature_type")

    def get_name(self):
        raise NotImplementedError("get_name")

    def batch_features(self, feature_list):
        features = np.stack(feature_list)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def get_features(self, state):
        raise NotImplementedError("get_features")


class MeanPooledImageFeatures(ImageFeatures):
    def __init__(self, image_feature_datasets):
        image_feature_datasets = sorted(image_feature_datasets)
        self.image_feature_datasets = image_feature_datasets

        self.mean_pooled_feature_stores = [paths.mean_pooled_feature_store_paths[dataset]
                                           for dataset in image_feature_datasets]
        self.feature_dim = MeanPooledImageFeatures.MEAN_POOLED_DIM * len(image_feature_datasets)
        print('Loading image features from %s' % ', '.join(self.mean_pooled_feature_stores))
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        self.features = defaultdict(list)
        for mpfs in self.mean_pooled_feature_stores:
            with open(mpfs, "rt") as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                for item in reader:
                    assert int(item['image_h']) == ImageFeatures.IMAGE_H
                    assert int(item['image_w']) == ImageFeatures.IMAGE_W
                    assert int(item['vfov']) == ImageFeatures.VFOV
                    long_id = self._make_id(item['scanId'], item['viewpointId'])
                    features = np.frombuffer(decode_base64(item['features']), dtype=np.float32).reshape(
                        (ImageFeatures.NUM_VIEWS, ImageFeatures.MEAN_POOLED_DIM))
                    self.features[long_id].append(features)
        assert all(len(feats) == len(self.mean_pooled_feature_stores) for feats in self.features.values())
        self.features = {
            long_id: np.concatenate(feats, axis=1)
            for long_id, feats in self.features.items()
        }

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def get_features(self, state):
        long_id = self._make_id(state['scanId'], state['viewpointId'])
        # Return feature of all the 36 views
        return self.features[long_id]

    def get_name(self):
        name = '+'.join(sorted(self.image_feature_datasets))
        name = "{}_mean_pooled".format(name)
        return name
