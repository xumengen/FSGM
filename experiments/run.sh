# train
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=0 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20 encoder='Linknet'
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=1 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20 encoder='Linknet'
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=2 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20 encoder='Linknet'
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=3 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20 encoder='Linknet'

# test
python test_proto.py with gpu_id=0 mode='test' snapshot='./runs/Metric_VOC_sets_0_1way_1shot_[train]_Linknet_MultiSimilarityLoss_MultiSimilarityMiner_20/1/snapshots/30000.pth' sample_num=20 encoder='Linknet'
python test_proto.py with gpu_id=0 mode='test' snapshot='./runs/Metric_VOC_sets_1_1way_1shot_[train]_Linknet_MultiSimilarityLoss_MultiSimilarityMiner_20/1/snapshots/30000.pth' sample_num=20 encoder='Linknet'
python test_proto.py with gpu_id=0 mode='test' snapshot='./runs/Metric_VOC_sets_2_1way_1shot_[train]_Linknet_MultiSimilarityLoss_MultiSimilarityMiner_20/1/snapshots/30000.pth' sample_num=20 encoder='Linknet'
python test_proto.py with gpu_id=0 mode='test' snapshot='./runs/Metric_VOC_sets_3_1way_1shot_[train]_Linknet_MultiSimilarityLoss_MultiSimilarityMiner_20/1/snapshots/30000.pth' sample_num=20 encoder='Linknet'
