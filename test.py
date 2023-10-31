"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
# from models.fewshot import *

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util import utils
from train import *
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from config import ex
from torchvision import transforms as transforms_torch
import os, sys
import matplotlib.pyplot as plt

# 创建的目录
path = "./CCG_Test"
if os.path.exists(path):
    # print("")
    pass
else:
    os.mkdir(path)


@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])  # 1234
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    # model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribblel_dilation']))
    transforms = Compose(transforms)

    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['n_steps'] * _config['batch_size'],
                # 每运行一次step，更新一次参数权重，即进行一次学习。每一次更新参数需要batch size个样本进行运算学习，根据运算结果调整更新一次参数
                # 训练和测试是要在同样条件下进行的，提出在训练的时候不断地让网络只看每一类的少量样本，这将和测试的过程是一致的
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries']
            )
            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")

            for sample_batched in tqdm.tqdm(testloader):
                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                # @czb  support_images_ids
                support_images_ids = [[shot for shot in way]
                                      for way in sample_batched['support_images_id']]
                # print(support_images_ids) @czb 测试不同模型的训练样本和验证样本采样是相同的
                suffix = 'scribble' if _config['scribble'] else 'mask'

                if _config['bbox']:
                    support_fg_mask = []
                    support_bg_mask = []
                    for i, way in enumerate(sample_batched['support_mask']):
                        fg_masks = []
                        bg_masks = []
                        for j, shot in enumerate(way):
                            fg_mask, bg_mask = get_bbox(shot['fg_mask'],
                                                        sample_batched['support_inst'][i][j])
                            fg_masks.append(fg_mask.float().cuda())
                            bg_masks.append(bg_mask.float().cuda())
                        support_fg_mask.append(fg_masks)
                        support_bg_mask.append(bg_masks)
                else:
                    support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]
                    support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                       for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]  # {1 3 447 447}
                query_images_id = sample_batched['query_images_id'][0]
                # print(query_images_id)

                query_labels = torch.cat(
                    [query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)  # 1 447 447


                query_pred, _, _, _ = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)  # 1 2 417 417 有值

                #  保存query image 和labels 11.26测试注释
                # utils.generalized_imshow(query_images[0].squeeze().cpu(), query_images_id[0])
                # utils.generalized_imshow(query_labels.cpu(), query_images_id[0] + "query_label")
                #
                # # @czb 保存预测结果1 447 447 shape
                # pred_1_447 = query_pred.argmax(dim=1)[0].unsqueeze(0)  # 1 447 447
                # utils.generalized_imshow(pred_1_447.cpu(), query_images_id[0] + "_argmax_pred")

                '''
                 @GL  合并预测和原图 
                 参数1： query images id
                 参数2： argmax拿到的 预测结果 1 447 447
                '''
                # utils.pred_query_cover(query_images_id[0], pred_1_447)  # 11.26 测试注释
                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),  # 预测最大的argmax为mask
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            # meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)
            # meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            # _run.log_scalar('classIoU', classIoU.tolist())
            # print('=========',classIoU.tolist())# mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            # _run.log_scalar('meanIoU', meanIoU.tolist())
            # print('=========', meanIoU.tolist())
            # _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            # _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            # print('++++++++++++', meanIoU_binary.tolist())
            # _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            # _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()
    _log.info('----- Final Result -----')
    # _run.log_scalar('final_classIoU', classIoU.tolist())
    # _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    # _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    # _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    # _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    # _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    # _log.info(f'classIoU mean: {classIoU}')
    # _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    # _log.info(f'meanIoU std: {meanIoU_std}')
    # _log.info(f'classIoU_binary mean: {classIoU_binary}')
    # _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    # _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')
