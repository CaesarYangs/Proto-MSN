#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from lib.extensions.parallel.data_container import DataContainer
from lib.utils.tools.logger import Logger as Log


def stack(batch, data_key=None, return_dc=False):
    if isinstance(batch[0][data_key], DataContainer):
        if batch[0][data_key].stack:
            assert isinstance(batch[0][data_key].data, torch.Tensor)
            samples = [sample[data_key].data for sample in batch]
            return default_collate(samples)

        elif not return_dc:
            return [sample[data_key].data for sample in batch]

        else:
            return DataContainer([sample[data_key].data for sample in batch])

    else:
        return default_collate([sample[data_key] for sample in batch])


def collate(batch, trans_dict):
    data_keys = batch[0].keys()
    if trans_dict['size_mode'] == 'diverse_size':
        target_widths = [batch[i]['img'].size(2) for i in range(len(batch))]
        target_heights = [batch[i]['img'].size(1) for i in range(len(batch))]

    elif trans_dict['size_mode'] == 'fix_size':
        target_width, target_height = trans_dict['input_size']
        target_widths, target_heights = [target_width] * len(batch), [target_height] * len(batch) 

    elif trans_dict['size_mode'] == 'multi_size':
        ms_input_size = trans_dict['ms_input_size']
        target_width, target_height = ms_input_size[random.randint(0, len(ms_input_size) - 1)]
        target_widths, target_heights = [target_width] * len(batch), [target_height] * len(batch) 

    elif trans_dict['size_mode'] == 'max_size':
        border_width = [sample['img'].size(2) for sample in batch]
        border_height = [sample['img'].size(1) for sample in batch]
        target_width, target_height = max(border_width), max(border_height)
        target_widths, target_heights = [target_width] * len(batch), [target_height] * len(batch) 

    else:
        raise NotImplementedError('Size Mode {} is invalid!'.format(trans_dict['size_mode']))

    if 'fit_stride' in trans_dict:
        stride = trans_dict['fit_stride']
        for i in range(len(batch)):
            target_width, target_height = target_widths[i], target_heights[i]
            pad_w = 0 if (target_width % stride == 0) else stride - (target_width % stride)  # right
            pad_h = 0 if (target_height % stride == 0) else stride - (target_height % stride)  # down
            target_widths[i] = target_width + pad_w
            target_heights[i] = target_height + pad_h

    for i in range(len(batch)):
        target_width, target_height = target_widths[i], target_heights[i]

        if 'meta' in data_keys:
            batch[i]['meta'].data['input_size'] = [target_width, target_height]

        channels, height, width = batch[i]['img'].size()
        if height == target_height and width == target_width:
            continue

        scaled_size = [width, height]
        # Log.info('{} {} {} {} {}'.format(target_width, target_height, scaled_size[0], scaled_size[1], trans_dict['align_method']))

        if trans_dict['align_method'] in ['only_scale', 'scale_and_pad']:
            w_scale_ratio = target_width / width
            h_scale_ratio = target_height / height
            if trans_dict['align_method'] == 'scale_and_pad':
                w_scale_ratio = min(w_scale_ratio, h_scale_ratio)
                h_scale_ratio = w_scale_ratio

            scaled_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
            if 'meta' in data_keys and 'border_size' in batch[i]['meta'].data:
                batch[i]['meta'].data['border_size'] = scaled_size

            scaled_size_hw = (scaled_size[1], scaled_size[0])
            batch[i]['img'] = DataContainer(F.interpolate(batch[i]['img'].data.unsqueeze(0),
                                            scaled_size_hw, mode='bilinear', align_corners=True).squeeze(0), stack=True)
            if 'labelmap' in data_keys:
                labelmap = batch[i]['labelmap'].data.unsqueeze(0).unsqueeze(0).float()
                labelmap = F.interpolate(labelmap, scaled_size_hw, mode='nearest').long().squeeze(0).squeeze(0)
                batch[i]['labelmap'] = DataContainer(labelmap, stack=True)

            if 'maskmap' in data_keys:
                maskmap = batch[i]['maskmap'].data.unsqueeze(0).unsqueeze(0).float()
                maskmap = F.interpolate(maskmap, scaled_size_hw, mode='nearest').long().squeeze(0).squeeze(0)
                batch[i]['maskmap'].data = DataContainer(maskmap, stack=True)

        pad_width = target_width - scaled_size[0]
        pad_height = target_height - scaled_size[1]
        
        # Log.info(f"target_width:{target_width},target_height:{target_height},scaled_size[0]:{scaled_size[0]},scaled_size[1]:{scaled_size[1]}")
        # Log.info(f"pad_width{pad_width},pad_height:{pad_height}")
        
        assert pad_height >= 0 and pad_width >= 0
        if pad_width > 0 or pad_height > 0:
            assert trans_dict['align_method'] in ['only_pad', 'scale_and_pad']
            left_pad = 0
            up_pad = 0
            if 'pad_mode' not in trans_dict or trans_dict['pad_mode'] == 'random':
                left_pad = random.randint(0, pad_width)  # pad_left
                up_pad = random.randint(0, pad_height)  # pad_up

            elif trans_dict['pad_mode'] == 'pad_left_up':
                left_pad = pad_width
                up_pad = pad_height

            elif trans_dict['pad_mode'] == 'pad_right_down':
                left_pad = 0
                up_pad = 0

            elif trans_dict['pad_mode'] == 'pad_center':
                left_pad = pad_width // 2
                up_pad = pad_height // 2

            elif trans_dict['pad_mode'] == 'pad_border':
                if random.randint(0, 1) == 0:
                    left_pad = pad_width
                    up_pad = pad_height
                else:
                    left_pad = 0
                    up_pad = 0
            else:
                Log.error('Invalid pad mode: {}'.format(trans_dict['pad_mode']))
                exit(1)

            pad = (left_pad, pad_width-left_pad, up_pad, pad_height-up_pad)

            batch[i]['img'] = DataContainer(F.pad(batch[i]['img'].data, pad=pad, value=0), stack=batch[i]['img'].stack)

            if 'labelmap' in data_keys:
                batch[i]['labelmap'] = DataContainer(F.pad(batch[i]['labelmap'].data, pad=pad, value=-1), stack=batch[i]['labelmap'].stack)

            if 'maskmap' in data_keys:
                batch[i]['maskmap'] = DataContainer(F.pad(batch[i]['maskmap'].data, pad=pad, value=0), stack=batch[i]['maskmap'].stack)

            if 'distance_map' in data_keys:
                batch[i]['distance_map'] = DataContainer(F.pad(batch[i]['distance_map'].data, pad=pad, value=255), stack=batch[i]['distance_map'].stack)

            if 'angle_map' in data_keys:
                batch[i]['angle_map'] = DataContainer(F.pad(batch[i]['angle_map'].data, pad=pad, value=0), stack=batch[i]['angle_map'].stack)
 
            if 'mask_label_map' in data_keys:
                batch[i]['mask_label_map'] = DataContainer(F.pad(batch[i]['mask_label_map'].data, pad=pad, value=-1), stack=batch[i]['mask_label_map'].stack)

            if 'direction_label_map' in data_keys:
                batch[i]['direction_label_map'] = DataContainer(F.pad(batch[i]['direction_label_map'].data, pad=pad, value=-1), stack=batch[i]['direction_label_map'].stack)

            if 'multi_label_direction_map' in data_keys:
                batch[i]['multi_label_direction_map'] = DataContainer(F.pad(batch[i]['multi_label_direction_map'].data, pad=pad, value=-1), stack=batch[i]['multi_label_direction_map'].stack)
                
            if 'energy_label_map' in data_keys:
                batch[i]['energy_label_map'] = DataContainer(F.pad(batch[i]['energy_label_map'].data, pad=pad, value=-1), stack=batch[i]['energy_label_map'].stack)

            if 'offsetmap_h' in data_keys:
                batch[i]['offsetmap_h'] = DataContainer(F.pad(batch[i]['offsetmap_h'].data, pad=pad, value=0), stack=batch[i]['offsetmap_h'].stack)

            if 'offsetmap_w' in data_keys:
                batch[i]['offsetmap_w'] = DataContainer(F.pad(batch[i]['offsetmap_w'].data, pad=pad, value=0), stack=batch[i]['offsetmap_w'].stack)

    return dict({key: stack(batch, data_key=key) for key in data_keys})





