# VOC 1-way 1-shot
for i in $(seq 90000 5000 100000)
do
python test.py with gpu_id=1 mode='test' snapshot=./runs/PANet_VOC_sets_0_1way_1shot_[train]_Metric_MultiSimilarityLoss/1/snapshots/$i.pth
python test.py with gpu_id=1 mode='test' snapshot=./runs/PANet_VOC_sets_1_1way_1shot_[train]_Metric_MultiSimilarityLoss/1/snapshots/$i.pth
python test.py with gpu_id=1 mode='test' snapshot=./runs/PANet_VOC_sets_2_1way_1shot_[train]_Metric_MultiSimilarityLoss/1/snapshots/$i.pth
done
