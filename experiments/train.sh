# VOC 1-way 1-shot
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=0 task.n_ways=1 task.n_shots=1
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=1 task.n_ways=1 task.n_shots=1
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=2 task.n_ways=1 task.n_shots=1
python train_metric_v2.py with gpu_id=0 mode='train' dataset='VOC' label_sets=3 task.n_ways=1 task.n_shots=1
