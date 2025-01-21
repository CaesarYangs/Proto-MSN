# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: RainbowSecret, JingyiXie, LangHuang
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2019
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
from thop import profile
from thop import clever_format
import wandb
from mmcv.cnn import get_model_complexity_info
import ptflops

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed


def mac_to_gflops(macs, batch_size=1, convert_to_gflops=True):
    # MACs to FLOPs: multiply by 2 (一个乘加操作通常被视为两次浮点运算)
    flops = macs * 2
    # 转换为每秒运算次数 (假设运行时间为1秒)
    flops_per_second = flops * batch_size
    # 转换为 GFLOPS
    gflops = flops_per_second / (10**9) if convert_to_gflops else flops_per_second
    return gflops


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """

    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        # self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)  # 初始化评估器

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None
        self.experiment = None

        self._init_model()

    def _init_model(self):
        # wandb Initialize logging section
        self.experiment = None
        if self.configer.get('wandb', 'use_wandb') is True:
            self.experiment = wandb.init(project='HRNet_Proto', resume='allow', anonymous='must', tags=[
                                         "Proto", "metrics", "custom_dataset_ce", "baseline"])

        self.seg_net = self.model_manager.semantic_segmentor(experiment=self.experiment)

        try:
            # init wandb model
            # set base training config
            if self.configer.get('wandb', 'use_wandb') is True:
                self.experiment.config = {'epochs': self.configer.get('solver', 'max_iters'), 'batch_size': self.configer.get('train', 'batch_size'), 'learning_rate': self.self.configer.get('lr', 'base_lr'), 'val_percent': self.configer.get(
                    'checkpoints', 'save_iters')/self.configer.get('solver', 'max_iters'), 'img_scale': self.configer.get('train', 'input_size'), 'amp': True, 'train_trans': self.congier.get('train_trans', 'trans_seq')}

                # set model information
                self.experiment.config = {'dataset': self.configer.get(
                    'dataset'), 'method': self.configer.get('method')}
                self.experiment.config = {
                    'backbone': self.configer.get('network', 'backbone')}

                # set protoseg paras
                self.experiment.config = {"protoseg": {"gamma": self.configer.get('protoseg', 'gamma'), "loss_ppc_weight": self.configer.get('protoseg', 'loss_ppc_weight'), "loss_ppd_weight": self.configer.get('protoseg', 'gamma'), "num_prototype": self.configer.get(
                    'protoseg', 'gamma'), "pretrain_prototype": self.configer.get('protoseg', 'gamma'), "use_rmi": self.configer.get('protoseg', 'gamma'), "use_prototype": self.configer.get('protoseg', 'gamma'), "update_prototype": self.configer.get('protoseg', 'gamma'), "warmup_iters": self.configer.get('protoseg', 'gamma')}}

        except:
            pass

        # show flops section
        if self.configer.get('running_settings', 'show_flops') == True:
            compute_method = 2
            if compute_method == 1:
                try:
                    print("checking model complexity info")
                    input = torch.randn(1, 3, 512, 512)
                    cmp_score, params = profile(self.seg_net, inputs=(input, ))
                    cmp_score, params = clever_format([cmp_score*2, params], "%.3f")
                    print(f"flops:{cmp_score}, params:{params}")

                except Exception as e:
                    Log.error(e)
                    print(self.seg_net)

            if compute_method == 2:
                try:
                    flops, params = get_model_complexity_info(
                        self.seg_net, (3, 512, 512))
                    split_line = '=' * 30
                    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
                        split_line, (3, 512, 512), flops, params))
                    print('!!!Please be cautious if you use the results in papers. '
                          'You may need to check if all ops are supported and verify that the '
                          'flops computation is correct.')
                except Exception as e:
                    Log.error(e)
            if compute_method == 3:
                try:
                    # 确保模型在正确的设备上
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.seg_net = self.seg_net.to(device)

                    # 计算 FLOPs 和参数数量
                    macs, params = ptflops.get_model_complexity_info(self.seg_net, (3, 512, 512),
                                                                     as_strings=True,
                                                                     print_per_layer_stat=False,
                                                                     verbose=True)
                    print(f"Computational complexity: {macs}")
                    # print(f"Computational complexity: {gflops:.2G} GFLOPS")
                    print(f"Number of parameters: {params}")

                    print('!!!Please be cautious if you use the results in papers. '
                          'You may need to check if all ops are supported and verify that the '
                          'flops computation is correct.')
                except Exception as e:
                    Log.error(e)

            exit(0)

        Log.info(f"network using:{self.configer.get('network', 'model_name')}")
        self.seg_net = self.module_runner.load_net(self.seg_net)    # consume function - load model from disk to memory

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader()
        self.pixel_loss = self.loss_manager.get_seg_loss(experiment=self.experiment)
        Log.info(f"loss using:{self.configer.get('loss', 'loss_type')}")
        # Log.info(f"num_subdomains:{self.configer.get('protoseg', 'num_subdomains')}")
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

        self.with_proto = True if self.configer.exists("protoseg") else False

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(
            group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(
            params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        fcn_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' in key:
                bb_lr.append(value)
            elif 'aux_layer' in key or 'upsample_proj' in key:
                fcn_lr.append(value)
            else:
                nbb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': fcn_lr, 'lr': self.configer.get('lr', 'base_lr') * 10},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()
        scaler = torch.amp.GradScaler()  # change it from `torch.cuda.amp.autocast()`

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get(
                'solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get(
                'solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for _, data_dict in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            self.optimizer.zero_grad()  # 每次BP前务必要将梯度清零

            if self.configer.get('lr', 'metric') == 'iters':
                # Log.info(f"current iter:{self.configer.get('iters')}")  # ! debug here
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'),
                    self.scheduler, self.optimizer, backbone_list=[0, ]
                )

            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()

            # ************** core section of training (with prototype or without prototype) **************
            with torch.amp.autocast("cuda"):  # change it from `torch.cuda.amp.autocast()`
                if not self.with_proto:
                    # *** non-prototype ***
                    outputs = self.seg_net(*inputs)
                else:
                    # *** prototype learning***
                    pretrain_prototype = True if self.configer.get(
                        'iters') < self. configer.get('protoseg', 'warmup_iters') else False
                    outputs = self.seg_net(*inputs, gt_semantic_seg=targets[:, None, ...],
                                           pretrain_prototype=pretrain_prototype)
            self.foward_time.update(time.time() - foward_start_time)

            # ***************** calculating loss *****************
            loss_start_time = time.time()
            if is_distributed():
                import torch.distributed as dist

                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that 
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp

                with torch.amp.autocast("cuda"):  # change it from `torch.cuda.amp.autocast()`
                    loss = self.pixel_loss(outputs, targets)
                    backward_loss = loss
                    display_loss = reduce_tensor(
                        backward_loss) / get_world_size()
            else:
                backward_loss = display_loss = self.pixel_loss(
                    outputs, targets)

            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()

            # backward_loss.backward()
            # self.optimizer.step()

            # torch.nn.utils.clip_grad_norm_(self.seg_net.parameters(), max_norm=1.0)
            scaler.scale(backward_loss).backward()
            scaler.step(self.optimizer)  # !debug:appears error when using lung dataset at epoch 714
            scaler.update()

            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
                    self.configer.get('iters') > normal_max_iters and \
                    ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or
                     self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
                self.optimizer.update_swa()

        self.configer.plus_one('epoch')

        self.display_training_section_results()

        # Check to val the current model.
        if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            Log.info(f"now counter: epoch :{self.configer.get('epoch') + 1} , \
                interval:{self.configer.get('solver', 'test_interval')}, indicator:{(self.configer.get('epoch') + 1) % self.configer.get('solver', 'test_interval')}")
            self.__val()

    def display_training_section_results(self):
        """
        Display epoch result
        """

        Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                 'Time {batch_time.sum:.3f}s / {2}epoch, ({batch_time.avg:.3f})\t'
                 'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                 'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                 'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                 'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                 'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                     self.configer.get(
                         'epoch'), self.configer.get('iters'),
                     self.configer.get('solver', 'display_iter'),
                     self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                     foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                     data_time=self.data_time, loss=self.train_losses))

        self.batch_time.reset()
        self.foward_time.reset()
        self.backward_time.reset()
        self.loss_time.reset()
        self.data_time.reset()
        self.train_losses.reset()

        #  if self.configer.get('wandb', 'use_wandb') is True:
        #         try:
        #             self.experiment.log({
        #                 "train_epoch": self.configer.get('epoch'),
        #                 "train_iteration": self.configer.get('iters'),
        #                 "display_iter": self.configer.get('solver', 'display_iter'),
        #                 "learning_rate": self.module_runner.get_lr(self.optimizer),
        #                 "batch_time_sum": self.batch_time.sum,
        #                 "batch_time_avg": self.batch_time.avg,
        #                 "forward_time_sum": self.foward_time.sum,
        #                 "forward_time_avg": self.foward_time.avg,
        #                 "backward_time_sum": self.backward_time.sum,
        #                 "backward_time_avg": self.backward_time.avg,
        #                 "loss_time_sum": self.loss_time.sum,
        #                 "loss_time_avg": self.loss_time.avg,
        #                 "data_load_time_sum": self.data_time.sum,
        #                 "data_load_time_avg": self.data_time.avg,
        #                 "current_loss": self.train_losses.val,
        #                 "average_loss": self.train_losses.avg
        #             })

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
          全部val部分 以框架形式书写
        """
        # & 1.将网络调整为评估模式
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()   # set device to eval mode (gpu) (distributed mode)

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:  # 每处理 10 个批次数据，进行同步并打印已处理图像数量的日志信息。
                if is_distributed():
                    dist.barrier()  # Synchronize all processes
                Log.info('{} images processed\n'.format(j))

            if self.configer.get('dataset') == 'lip':
                (inputs, targets, inputs_rev, targets_rev), batch_size = self.data_helper.prepare_data(data_dict,
                                                                                                       want_reverse=True)
            else:
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            with torch.no_grad():
                if self.configer.get('dataset') == 'lip':  # 目前完全不牵扯到这段代码内容
                    inputs = torch.cat([inputs[0], inputs_rev[0]], dim=0)
                    outputs = self.seg_net(inputs)
                    if not is_distributed():
                        outputs_ = self.module_runner.gather(outputs)
                    else:
                        outputs_ = outputs
                    if isinstance(outputs_, (list, tuple)):
                        outputs_ = outputs_[-1]
                    outputs = outputs_[
                        0:int(outputs_.size(0) / 2), :, :, :].clone()
                    outputs_rev = outputs_[
                        int(outputs_.size(0) / 2):int(outputs_.size(0)), :, :, :].clone()
                    if outputs_rev.shape[1] == 20:
                        outputs_rev[:, 14, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 15, :, :]
                        outputs_rev[:, 15, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 14, :, :]
                        outputs_rev[:, 16, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 17, :, :]
                        outputs_rev[:, 17, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 16, :, :]
                        outputs_rev[:, 18, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 19, :, :]
                        outputs_rev[:, 19, :, :] = outputs_[
                            int(outputs_.size(0) / 2):int(outputs_.size(0)), 18, :, :]
                    outputs_rev = torch.flip(outputs_rev, [3])
                    outputs = (outputs + outputs_rev) / 2.
                    self.evaluator.update_score(
                        outputs, data_dict['meta'])

                elif self.data_helper.conditions.diverse_size:  # 针对不同尺寸的输入进行并行处理-目前完全不牵扯到这段代码内容
                    if is_distributed():
                        outputs = [self.seg_net(inputs[i])
                                   for i in range(len(inputs))]
                    else:
                        outputs = nn.parallel.parallel_apply(
                            replicas[:len(inputs)], inputs)

                    for i in range(len(outputs)):
                        loss = self.pixel_loss(
                            outputs[i], targets[i].unsqueeze(0))
                        # self.val_losses.update(loss.item(), 1)
                        outputs_i = outputs[i]
                        if isinstance(outputs_i, torch.Tensor):
                            outputs_i = [outputs_i]
                        self.evaluator.update_score(
                            outputs_i, data_dict['meta'][i:i + 1])

                else:
                    # & 2.使用分割网络进行预测
                    outputs = self.seg_net(*inputs)  # inputs: 星号（*）运算符将 inputs 变量（可能是列表或元组）解包为多个张量

                    if not is_distributed():
                        outputs = self.module_runner.gather(outputs)
                    if isinstance(outputs, dict):
                        outputs = outputs['seg']

                    # outputs: {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}
                    self.evaluator.update_score(outputs, data_dict['meta'])  # calculate performance score

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        # & 3.根据配置更新评估指标，保存网络模型，并打印评估结果
        curr_perf, max_perf, perf_type = self.evaluator.update_performance()

        self.module_runner.save_net(self.seg_net, save_mode='performance')
        cudnn.benchmark = True

        # Print the log info & reset the states.
        self.evaluator.reduce_scores()  # 存储聚合后的最终混淆矩阵
        if not is_distributed() or get_rank() == 0:
            metrics_results_dic = self.evaluator.print_scores()   # 打印全部得分信息

        self.batch_time.reset()
        self.evaluator.reset()

        # & 4.将网络改回训练模式
        self.seg_net.train()
        self.pixel_loss.train()

        # & 5.上传所有数据到wandb
        # outputs, curr_perf, max_perf, perf_type, confusion_matrix_np, metrics_results_dic
        if self.experiment is not None and self.configer.get('wandb', 'use_wandb') is True:
            self.experiment.log({
                "performance": curr_perf,
                "max_performance": max_perf,
                "metric_name": perf_type,
                "Mean IOU": metrics_results_dic['Mean IOU'],
                "Pixel ACC": metrics_results_dic['Pixel ACC'],
                "F1 Score": metrics_results_dic['F1 Score'],
                "Precision": metrics_results_dic['Precision'],
                "Recall": metrics_results_dic['Recall']
            }, step=self.configer.get('iters'))  # 使用 iters 作为 step

    def train(self):
        # cudnn.benchmark = True
        # self.__val()
        if self.configer.get('network', 'resume') is not None:
            if self.configer.get('network', 'resume_val'):
                self.__val(
                    data_loader=self.data_loader.get_valloader(dataset='val'))
                return
            elif self.configer.get('network', 'resume_train'):
                self.__val(
                    data_loader=self.data_loader.get_valloader(dataset='train'))
                return
            # return

        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
            return

        while self.configer.get('epoch') < self.configer.get('solver', 'max_epoch'):    # change here to epoch as main conductor
            self.__train()

        # use swa to average the model, final validation on the last round
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

        Log.critical(f"training process end in {self.configer.get('epoch')} epochs")

    def summary(self):
        from lib.utils.summary import get_model_summary
        import torch.nn.functional as F
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
