import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from lib.utils.tools.logger import Logger as Log
from lib.metrics import running_score as rslib
from lib.metrics import F1_running_score as fscore_rslib
from lib.utils.distributed import get_world_size, get_rank, is_distributed


class _BaseEvaluator:

    def __init__(self, configer, trainer):
        self.configer = configer
        self.trainer = trainer
        self._init_running_scores()
        self.conditions = configer.conditions

    def use_me(self):
        raise NotImplementedError

    def _init_running_scores(self):
        raise NotImplementedError

    def update_score(self, *args, **kwargs):
        raise NotImplementedError

    def print_scores(self, show_miou=True):
        results = {}
        for key, rs in self.running_scores.items():
            Log.info('Result for {}'.format(key))
            if isinstance(rs, fscore_rslib.F1RunningScore):
                FScore, FScore_cls = rs.get_scores()
                Log.info('Mean FScore: {}'.format(FScore))
                Log.info(
                    'Class-wise FScore: {}'.format(
                        ', '.join(
                            '{:.3f}'.format(x)
                            for x in FScore_cls
                        )
                    )
                )
                results['Mean FScore'] = FScore
                results['Class-wise FScore'] = FScore_cls
                results[key] = {
                    'Mean FScore': FScore,
                    'Class-wise FScore': FScore_cls,
                }
            elif isinstance(rs, rslib.SimpleCounterRunningScore):
                Log.info('ACC: {}\n'.format(rs.get_mean_acc()))
                results[key] = {'ACC': rs.get_mean_acc()}
            else:
                if show_miou and hasattr(rs, 'get_mean_iou'):
                    Log.info('Mean IOU: {}\n'.format(rs.get_mean_iou()))
                Log.info('Pixel ACC: {}\n'.format(rs.get_pixel_acc()))

                if hasattr(rs, 'n_classes') and rs.n_classes == 2:
                    Log.info(
                        'F1 Score: {} Precision: {} Recall: {}\n'
                        .format(*rs.get_F1_score())
                    )
                results['Mean IOU'] = rs.get_mean_iou()
                results['Pixel ACC'] = rs.get_pixel_acc()
                results['F1 Score'] = rs.get_F1_score()[0]
                results['Precision'] = rs.get_F1_score()[1]
                results['Recall'] = rs.get_F1_score()[2]

        return results   

    def prepare_validaton(self):
        """
        Replicate models if using diverse size validation.
        """
        if is_distributed():
            return
        device_ids = list(range(len(self.configer.get('gpu'))))
        if self.conditions.diverse_size:
            cudnn.benchmark = False
            assert self.configer.get('val', 'batch_size') <= len(device_ids)
            replicas = nn.parallel.replicate(
                self.trainer.seg_net.module, device_ids)
            return replicas

    def update_performance(self,experiment = None):

        try:
            rs = self.running_scores[self.save_net_main_key]    # 当前使用的一般为SegTask, 对应到使用RunningScore, 计算miou
            if self.save_net_metric == 'miou':  # 一般都会获取这个
                perf = rs.get_mean_iou()
            elif self.save_net_metric == 'acc':
                perf = rs.get_pixel_acc()

            # 更新max perf, 仅此而已 这块没有别的功能
            max_perf = self.configer.get('max_performance')
            self.configer.update(['performance'], perf)
            if perf > max_perf and (not is_distributed() or get_rank() == 0):
                Log.info('Performance {} -> {}'.format(max_perf, perf))
                
            return perf, max_perf, self.save_net_metric
        
        except Exception as e:
            Log.warn(e)

    def reset(self):
        for rs in self.running_scores.values():
            rs.reset()

    def reduce_scores(self):
        for rs in self.running_scores.values():
            if hasattr(rs, 'reduce_scores'):
                rs.reduce_scores()
        
        Log.info("print confusion matrix")
        self.running_scores['seg'].print_confusion_matrix()
