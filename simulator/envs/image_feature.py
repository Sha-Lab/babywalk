import numpy as np
import sys
import paths
import csv

from collections import defaultdict
from envs_utils import decode_base64

csv.field_size_limit(sys.maxsize)


class ImageFeatures(object):
  NUM_VIEWS = 36
  MEAN_POOLED_DIM = 2048
  IMAGE_W = 640
  IMAGE_H = 480
  VFOV = 60
  
  @staticmethod
  def from_args(args):
    for image_feature_type in sorted(args.image_feature_type):
      assert image_feature_type == "mean_pooled"
    return [MeanPooledImageFeatures(args.image_feature_datasets)]
  
  @staticmethod
  def add_args(args):
    args.add_argument("--num_views", type=int,
                      default=ImageFeatures.NUM_VIEWS)
    args.add_argument("--mean_pooled_dim", type=int,
                      default=ImageFeatures.MEAN_POOLED_DIM)
    args.add_argument("--image_feature_type", nargs="+",
                      default=["mean_pooled"])
    args.add_argument("--image_feature_datasets", nargs="+",
                      default=["imagenet"])
  
  def get_features(self, state):
    raise NotImplementedError("get_features")


class MeanPooledImageFeatures(ImageFeatures):
  def __init__(self, image_feature_datasets):
    image_feature_datasets = sorted(image_feature_datasets)
    self.image_feature_datasets = image_feature_datasets
    self.mean_pooled_feature_stores = [
      paths.MEAN_POOLED_FEATURE_STORE_PATHS[dataset]
      for dataset in image_feature_datasets]
    self.feature_dim = MeanPooledImageFeatures.MEAN_POOLED_DIM \
                       * len(image_feature_datasets)
    print('Loading image features from %s'
          % ', '.join(self.mean_pooled_feature_stores))
    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov',
                      'features']
    self.features = defaultdict(list)
    for mpfs in self.mean_pooled_feature_stores:
      with open(mpfs, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                fieldnames=tsv_fieldnames)
        for item in reader:
          assert int(item['image_h']) == ImageFeatures.IMAGE_H
          assert int(item['image_w']) == ImageFeatures.IMAGE_W
          assert int(item['vfov']) == ImageFeatures.VFOV
          long_id = self._make_id(item['scanId'], item['viewpointId'])
          features = np.frombuffer(decode_base64(item['features']),
                                   dtype=np.float32)
          features = features.reshape((ImageFeatures.NUM_VIEWS,
                                       ImageFeatures.MEAN_POOLED_DIM))
          self.features[long_id].append(features)
    assert all(
      len(feats) == len(self.mean_pooled_feature_stores)
      for feats in self.features.values()
    )
    self.features = {
      long_id: np.concatenate(feats, axis=1)
      for long_id, feats in self.features.items()
    }
  
  def _make_id(self, scan_id, viewpoint_id):
    return scan_id + '_' + viewpoint_id
  
  def get_features(self, state):
    long_id = self._make_id(state['scan_id'], state['viewpoint_id'])
    return self.features[long_id]
