"""Training Script"""
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity

import random


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
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], config=_config)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
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
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    
    # loss func
    assert _config['loss'] in ['MultiSimilarityLoss', 'ContrastiveLoss', 'ProxyNCALoss', 'ProxyAnchorLoss']
    if _config['loss'] == 'MultiSimilarityLoss':
        loss_func = losses.MultiSimilarityLoss()
    elif _config['loss'] == 'ContrastiveLoss':
        loss_func = losses.ContrastiveLoss()
    elif _config['loss'] == 'ProxyNCALoss':
        loss_func = losses.ProxyNCALoss(num_classes=2, embedding_size=_config['output_feature_length'])
    elif _config['loss'] == 'ProxyAnchorLoss':
        loss_func = losses.ProxyAnchorLoss()

    # miner
    if _config['miner'] == 'MultiSimilarityMiner':
        miner = miners.MultiSimilarityMiner()
    elif _config['miner'] == 'PairMarginMiner':
        miner = miners.PairMarginMiner()
    elif _config['miner'] == 'MaximumLossMiner':
        miner = miner.MaximumLossMiner()

    i_iter = 0
    log_loss = {'loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        supp_fg_fts, supp_bg_fts, qry_fts = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images)
        
        # Metric learning
        supp_fg_fts_list = [[] for _ in range(len(supp_fg_fts))]
        supp_bg_fts_list = []
        try:
            for i in range(len(supp_fg_fts)):
                for j in range(len(supp_fg_fts[0])):
                    supp_fg_fts_list[i].append(supp_fg_fts[i][j].squeeze().transpose(0, 1))
                    supp_bg_fts_list.append(supp_bg_fts[i][j].squeeze().transpose(0, 1))
            
            # supp fg fts
            supp_fg_fts_tensor = []
            for supp_fg_fts in supp_fg_fts_list:
                supp_fg_fts_tensor.append(torch.cat(supp_fg_fts, dim=0))
            # supp bg fts
            supp_bg_fts_tensor = torch.cat(supp_bg_fts_list, dim=0)
            # supp fts
            supp_fts_list = [supp_bg_fts_tensor] + supp_fg_fts_tensor
        except:
            continue

        # extract foreground and background query features
        qry_fts = F.interpolate(qry_fts[0], size=query_labels.shape[-2:], mode='bilinear')  # 1 * C * H * W
        label_num = torch.unique(query_labels)
        qry_fts_list = []
        for i in range(len(supp_fts_list)):
            index = torch.where(query_labels==i)
            qry_fts_num = qry_fts[:, :, index[-2], index[-1]]  # 1 * C * N1'
            qry_fts_list.append(qry_fts_num[0].transpose(0, 1))  # N1' * C

        # concat fts
        concat_fts = []
        for i in range(len(supp_fts_list)):
            concat_fts.append(torch.cat((supp_fts_list[i], qry_fts_list[i]), dim=0))
        
        # sample the features
        N_nums = []
        for i in range(len(concat_fts)):
            N_nums.append(concat_fts[i].shape[0])

        if not np.array(N_nums).all():
            continue

        # compute loss
        fts_sample = []
        labels_sample = []
        for i in range(len(concat_fts)):
            k = _config['sample_num'] if N_nums[i] >= _config['sample_num'] else N_nums[i]
            indices = torch.tensor(random.sample(range(N_nums[i]), k))
            fts_sample.append(concat_fts[i][indices])
            labels_sample.append(torch.full((k,), i))

        fts = torch.cat(fts_sample, dim=0)
        label = torch.cat(labels_sample)

        hard_pairs = miner(fts, label)
        loss = loss_func(fts, label.cuda(), hard_pairs)

        try:
            loss.backward()
            optimizer.step()
            scheduler.step()
        except:
            continue

        # Log loss
        loss = loss.detach().data.cpu().numpy()
        _run.log_scalar('loss', loss)
        log_loss['loss'] += loss


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
