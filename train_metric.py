"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot_v4 import FewShotSeg
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
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
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
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    # loss_func = losses.ContrastiveLoss(distance = CosineSimilarity())
    # loss_func = losses.CircleLoss(distance = CosineSimilarity())
    miner = miners.MultiSimilarityMiner()
    # loss_func = losses.MultiSimilarityLoss()
    # loss_func = losses.NCALoss()
    # loss_func = losses.NormalizedSoftmaxLoss(num_classes=2, embedding_size=512)
    # main_loss = losses.TupletMarginLoss()
    # var_loss = losses.IntraPairVarianceLoss()
    # loss_func = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
    loss_func = losses.ContrastiveLoss()

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
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
        supp_fg_fts_list = []
        supp_bg_fts_list = []
        for j in range(len(supp_fg_fts[0])):
            supp_fg_fts_list.append(supp_fg_fts[0][j].squeeze().transpose(0, 1))
            supp_bg_fts_list.append(supp_bg_fts[0][j].squeeze().transpose(0, 1))
        supp_fg_fts = torch.cat(supp_fg_fts_list, dim=0)  # N1 * C
        supp_bg_fts = torch.cat(supp_bg_fts_list, dim=0)  # N2 * C

        # supp_fg_fts = supp_fg_fts[0][0].squeeze().transpose(0, 1)  # N1 * C
        # supp_bg_fts = supp_bg_fts[0][0].squeeze().transpose(0, 1)  # N2 * C

        # extract foreground and background query features
        qry_fts = F.interpolate(qry_fts[0], size=query_labels.shape[-2:], mode='bilinear')  # 1 * C * H * W
        fore_index = torch.where(query_labels==1)
        back_index = torch.where(query_labels==0)
        qry_fore_fts = qry_fts[:, :, fore_index[-2], fore_index[-1]]  # 1 * C * N1'
        qry_back_fts = qry_fts[:, :, back_index[-2], back_index[-1]]  # 1 * C * N2'
        qry_fg_fts = qry_fore_fts[0].transpose(0, 1)  # N1' * C
        qry_bg_fts = qry_back_fts[0].transpose(0, 1)  # N2' * C

        # concat fts
        fg_fts = torch.cat((supp_fg_fts, qry_fg_fts), dim=0)
        bg_fts = torch.cat((supp_bg_fts, qry_bg_fts), dim=0)
        
        # sample the features
        Nf, C = fg_fts.shape
        Nb, C = bg_fts.shape

        if not Nf or not Nb:
            continue

        # compute loss
        k = 5000 if Nf >= 5000 else Nf
        indices = torch.tensor(random.sample(range(Nf), k))
        fg_fts_sample = fg_fts[indices]
        fg_label_sample = torch.full((k, ), 1)

        k = 5000 if Nb >= 5000 else Nb
        indices = torch.tensor(random.sample(range(Nb), k))
        bg_fts_sample = bg_fts[indices]
        bg_label_sample = torch.full((k, ), 0)

        fts = torch.cat((fg_fts_sample, bg_fts_sample), dim=0)  # 6000 * C
        label = torch.cat((fg_label_sample, bg_label_sample))  # 6000
        # fts = torch.cat((qry_fg_fts, qry_bg_fts), dim=0)
        # label = torch.cat((qry_fg_label, qry_bg_label))

        hard_pairs = miner(fts, label)
        loss = loss_func(fts, label.cuda(), hard_pairs)
        
        # loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = 0
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
