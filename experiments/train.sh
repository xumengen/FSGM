# VOC 1-way 1-shot
python train_metric.py with gpu_id=1 mode='train' dataset='VOC' label_sets=0 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20 
python train_metric.py with gpu_id=1 mode='train' dataset='VOC' label_sets=1 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20
python train_metric.py with gpu_id=1 mode='train' dataset='VOC' label_sets=2 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20
python train_metric.py with gpu_id=1 mode='train' dataset='VOC' label_sets=3 task.n_ways=1 task.n_shots=1 loss='MultiSimilarityLoss' sample_num=20
