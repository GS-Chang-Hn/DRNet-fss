"""Training Script"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex
from tensorboardX import SummaryWriter
from models import trimodal_attention as att_fusion
import math
import torch.nn.functional as F

writer = SummaryWriter('./results/scalar_example')


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    model.train()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],  # ./data/Pascal/VOCdevkit/VOC2012/
        split=_config['path'][data_name]['data_split'],  # trainaug
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,  # {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        max_iters=_config['n_steps'] * _config['batch_size'],  # 50000 * 1
        n_ways=_config['task']['n_ways'],  # 1
        n_shots=_config['task']['n_shots'],  # 1
        n_queries=_config['task']['n_queries']  # 1
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    # optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion1 = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    criterion2 = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    criterion3 = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    criterion4 = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    c1_t = torch.tensor(torch.rand(1), requires_grad=True).to(device)
    c1_f = torch.tensor(1 - c1_t, requires_grad=True).to(device)

    q_f_w = torch.tensor(torch.rand(1), requires_grad=True).to(device)
    q_b_w = torch.tensor(torch.rand(1), requires_grad=True).to(device)
    s_b_w = torch.tensor(torch.rand(1), requires_grad=True).to(device)
    weights = torch.cat((q_f_w, q_b_w, s_b_w), dim=0)
    weights = F.softmax(weights, dim=0)
    q_f_w = weights[0]
    q_b_w = weights[1]
    s_b_w = weights[2]
    s_f_w = torch.tensor(1 - q_f_w - q_b_w - s_b_w, requires_grad=True).to(device)
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        # print(sample_batched['support_images'][0][0].shape)  # [1, 3, 417, 417]  1 3 390 390
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_labels = torch.stack(
            [torch.stack([support_lab.long().cuda() for support_lab in way]) for way in support_bg_mask],
            dim=0).squeeze(0).squeeze(1)

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        que_output_foreground, que_output_background, sup_output_foreground, sup_output_background = model(
            support_images, support_fg_mask, support_bg_mask,
            query_images)

        query_loss_true = criterion1(que_output_foreground, query_labels)
        query_loss_false = criterion2(que_output_background, query_labels)
        sup_loss_true = criterion3(sup_output_foreground, support_labels)
        sup_loss_false = criterion4(sup_output_background, support_labels)
        # loss = s_f_w * math.exp(c1_t * sup_loss_true) + s_b_w * math.exp(-c1_f * sup_loss_false) + q_f_w * math.exp(
        #     c1_t * query_loss_true) + q_b_w * math.exp(-c1_f * query_loss_false)
        loss = q_f_w * query_loss_true + s_f_w * sup_loss_true
        # loss = query_loss_true
        # uery_loss_true
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Log loss
        loss = loss.detach().data.cpu().numpy()
        # align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', loss)
        # _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += loss
        # log_loss['align_loss'] += align_loss
        # 保存loss到log 可视化
        writer.add_scalar('Total_loss', loss, global_step=i_iter)
        writer.add_scalar('query_loss_true', query_loss_true, global_step=i_iter)
        # writer.add_scalar('sup_loss_false', sup_loss_false, global_step=i_iter)
        # writer.add_scalar('Query-Support', align_loss, global_step=i_iter)
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            # align_loss = log_loss['align_loss'] / (i_iter + 1)
            # print(
            #     f'step {i_iter + 1}: loss: {loss}, q_f_w: {query_loss_true}, q_b_w: {query_loss_false}, s_f_w: {sup_loss_true}, s_b_w: {sup_loss_false}, c1_t: {c1_t}, c1_f: {c1_f}')
            print(f'step {i_iter + 1}: loss: {loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
