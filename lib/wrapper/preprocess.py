import torch
import numpy as np
from lib.utils.geo_transform import axis_angle_to_rot_6d
from lib.utils.registry import Registry
import clip
import json
import pickle
PRE = Registry('preprocess')


@PRE.register()
def process_clip_text(wrapper, batch):
    utterence = [meta['utterance'] for meta in batch['meta']]
    text_token = clip.tokenize(utterence).to(batch['motion_mask'].device)
    text_feature = wrapper.clip_model.encode_text(text_token)
    batch.update({
        'text_feature': text_feature,
    })