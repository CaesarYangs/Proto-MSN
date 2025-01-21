import os

import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import Counter
import wandb

from lib.utils.tools.logger import Logger as Log
from .base import _BaseEvaluator
from . import tasks

def _parse_output_spec(spec):
    """
    Parse string like "mask, _, dir, ..., seg" into indices mapping
    {
        "mask": 0,
        "dir": 2,
        "seg": -1
    }
    """
    spec = [x.strip() for x in spec.split(',')]
    existing_task_names = set(tasks.task_mapping)

    # `spec` should not have invalid keys other than in `existing_task_names`
    assert set(spec) - ({'...', '_'} | existing_task_names) == set()
    # `spec` should have at least one key in `existing_task_names`
    assert set(spec) & existing_task_names != set()

    counter = Counter(spec)
    for task in tasks.task_mapping.values():
        task.validate_output_spec(spec, counter)
    assert counter['...'] <= 1

    length = len(spec)
    output_indices = {}
    negative_index = False
    for idx, name in enumerate(spec):
        if name not in ['_', '...']:
            index = idx - length if negative_index else idx
            output_indices[name] = index
        elif name == '...':
            negative_index = True

    return output_indices


class StandardEvaluator(_BaseEvaluator):
    """一般都是使用这个作为核心Evaluator

    Args:
        _BaseEvaluator (_type_): _description_
    """

    def _output_spec(self):
        if self.configer.conditions.pred_dt_offset:
            default_spec = 'mask, dir'
        elif self.configer.conditions.pred_ml_dt_offset:
            default_spec = 'mask, ml_dir'
        else:
            default_spec = '..., seg'
            
        Log.info(f"StandardEvaluator spec using:{default_spec}")

        return os.environ.get('output_spec', default_spec)

    def _init_running_scores(self):
        self.output_indices = _parse_output_spec(self._output_spec())

        self.running_scores = {}
        for task in tasks.task_mapping.values():
            # 初始化为标准RunningScore(SegTask情况)
            rss, main_key, metric = task.running_score(self.output_indices, self.configer)
            if rss is None:
                continue
            self.running_scores.update(rss)
            self.save_net_main_key = main_key
            self.save_net_metric = metric

    def update_score(self, outputs, metas,experiment = None):
        confusion_matrix = []
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        
        # outputs[0]: torch.Size([4, 2, 128, 128])
        for i in range(len(outputs[0])): # per batch
            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']

            outputs_numpy = {}
            for name, idx in self.output_indices.items():
                item = outputs[idx].permute(0, 2, 3, 1)
                if self.configer.get('dataset') == 'celeba':
                    # the celeba image is of size 1024x1024 - 目前不会运行到这个地方
                    item = cv2.resize(
                        item[i, :border_size[1], :border_size[0]].cpu().numpy(),
                        tuple(x // 2 for x in ori_img_size), interpolation=cv2.INTER_CUBIC
                    )
                else:
                    # transfer val img from feature size to original image (per image using plus)
                    item = cv2.resize(
                        item[i, :border_size[1], :border_size[0]].cpu().numpy(),
                        tuple(ori_img_size), interpolation=cv2.INTER_CUBIC
                    )
                    # item = 1 - item # TODO: test for hardcode and short path here
                    
                outputs_numpy[name] = item

            # 核心评估函数
            for name in outputs_numpy:
                tasks.task_mapping[name].eval(
                    outputs_numpy, metas[i], self.running_scores
                )

    def print_confusion_matrix(self):
        Log.info("print confusion matrix")
        self.running_scores['seg'].print_confusion_matrix()