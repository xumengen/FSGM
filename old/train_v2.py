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

from models.fewshot_v2 import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex

from pytorch_metric_learning import miners, losses


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
    # criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=3)
    miner = miners.MultiSimilarityMiner()

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
        supp_fts, qry_fts = model(support_images, query_images)
        channel_size = qry_fts.size()[1]

        # Support Metric Learning
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in support_fg_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in support_bg_mask], dim=0)  # Wa x Sh x B x H x W
        class_num = fore_mask.size()[0]

        new_supp_mask = torch.zeros_like(fore_mask)
        for idx, mask in enumerate(fore_mask):
            new_supp_mask[idx] = torch.where(mask==1, idx+1, 0)

        supp_fts = F.interpolate(supp_fts, size=new_supp_mask.size()[-2:], mode='bilinear').reshape(-1, channel_size)  # N(Wa*Sh*B*H*W) * C
        support_labels = new_supp_mask.reshape(-1)  # N

        # Query Metric Learning
        qry_fts = F.interpolate(qry_fts, size=query_labels.size()[-2:], mode='bilinear').reshape(-1, channel_size)
        query_labels = query_labels.reshape(-1)

        positive_labels = torch.nonzero(torch.eq(support_labels, 1))
        positive_indices = torch.randperm(len(positive_labels))[:2000]
        negative_labels = torch.nonzero(torch.eq(support_labels, 0))
        negative_indices = torch.randperm(len(negative_labels))[:1000]
        indices = torch.cat((positive_indices, negative_indices))
        supp_fts = supp_fts[indices]
        support_labels = support_labels[indices]
        # indices = torch.randperm(len(support_labels))[:200]
        # hard_pairs = miner(supp_fts[indices], support_labels[indices])     
        # support_loss = loss_func(supp_fts[indices], support_labels[indices], hard_pairs)  
        # supp_positive_mean = torch.mean(supp_fts[positive_labels], dim=0, keepdim=True) 
        # supp_negative_mean = torch.mean(supp_fts[negative_labels], dim=0, keepdim=True)
        # supp_fts = torch.cat((supp_negative_mean, supp_positive_mean), dim=0)


        positive_labels = torch.nonzero(torch.eq(query_labels, 1))
        positive_indices = torch.randperm(len(positive_labels))[:2000]
        negative_labels = torch.nonzero(torch.eq(query_labels, 0))
        negative_indices = torch.randperm(len(negative_labels))[:1000]
        indices = torch.cat((positive_indices, negative_indices))
        qry_fts = qry_fts[indices]
        query_labels = query_labels[indices]
        # indices = torch.randperm(len(query_labels))[:200]
        # hard_pairs = miner(qry_fts[indices], query_labels[indices])     
        # query_loss = loss_func(qry_fts[indices], query_labels[indices], hard_pairs) 
        # qry_positive_mean = torch.mean(qry_fts[positive_labels], dim=0, keepdim=True) 
        # qry_negative_mean = torch.mean(qry_fts[negative_labels], dim=0, keepdim=True)
        # qry_fts = torch.cat((qry_negative_mean, qry_positive_mean), dim=0)

        # pdist = nn.PairwiseDistance(p=2)
        # dist_pos_pos = pdist(supp_positive_mean, qry_positive_mean)[0]
        # dist_neg_neg = pdist(supp_negative_mean, qry_negative_mean)[0]
        # dist_pos_neg = pdist(supp_positive_mean, qry_negative_mean)[0]
        # dist_neg_pos = pdist(supp_negative_mean, qry_positive_mean)[0]
        # print(dist_pos_pos, dist_neg_neg, dist_pos_neg, dist_neg_pos)
        # loss = dist_pos_pos + dist_neg_neg - dist_pos_neg - dist_neg_pos

        
        fts = torch.cat((supp_fts, qry_fts), dim=0)
        labels = torch.cat((support_labels, query_labels), dim=0)
        # labels = torch.tensor([0, 1, 0, 1])
        hard_pairs = miner(fts, labels)
        loss = loss_func(fts, labels, hard_pairs) 

        # Loss backward
        align_loss = 0 
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        loss = loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += loss
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
