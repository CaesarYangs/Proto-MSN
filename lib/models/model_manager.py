# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Microsoft Research
# Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
# Copyright (c) 2019
# yuyua@microsoft.com
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# HiFormer
from lib.models.nets.ProtoMSN import HiFormer, HiFormer_PROTO_MED_MAX

from lib.utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    'HiFormer': HiFormer,
    'HiFormer_PROTO_MAX': HiFormer_PROTO_MED_MAX,
}


class ModelManager(object):
    def __init__(self, configer, experiment=None):
        self.configer = configer
        self.experiment = experiment

    def semantic_segmentor(self, experiment=None):
        model_name = self.configer.get('network', 'model_name')
        self.experiment = experiment

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        # model = SEG_MODEL_DICT[model_name](self.configer,experiment=self.experiment)
        model = SEG_MODEL_DICT[model_name](self.configer)

        return model

    def instance_segmentor(self, experiment=None):
        # TODO: add your instance segmentation method here
        pass
