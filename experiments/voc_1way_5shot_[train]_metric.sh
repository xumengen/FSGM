# VOC 1-way 1-shot
python train_metric.py with gpu_id=0 mode='train' dataset='VOC' label_sets=0 model.align=False task.n_ways=1 task.n_shots=5
python train_metric.py with gpu_id=0 mode='train' dataset='VOC' label_sets=1 model.align=False task.n_ways=1 task.n_shots=5
python train_metric.py with gpu_id=0 mode='train' dataset='VOC' label_sets=2 model.align=False task.n_ways=1 task.n_shots=5
python train_metric.py with gpu_id=0 mode='train' dataset='VOC' label_sets=3 model.align=False task.n_ways=1 task.n_shots=5
