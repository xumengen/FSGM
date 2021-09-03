# Apply Deep Metric Learning to Few-Shot Image Segmentation

<!-- ABOUT THE PROJECT -->
## About The Project
In this repo, deep metric learning is applied to few shot segmentation. According to the information provided by the mask, the distance between feature vectors of the same category in the high-dimensional space is as small as possible, and the distance between feature vectors of different categories is close or far away.


<!-- GETTING STARTED -->
## Getting Started

### Folder Structure
* dataloaders/ 
  * `coco.py`, `common.py`, `customized.py` are the class of dataset and dataloaders.
  * `transforms.py` contains different transform functions.
  
* models/
  * `fewshot.py` is the class definition of train model, including model initialization and forward function.
  * `few_proto.py` is the class definition of test prototype model.
  * `vgg.py` is the class definition of VGG feature extractor.

* util/ 
  * `metric.py` is the metric functions to evaluate models.
  * `utils.py` contains some other functions used in the model training and testing.

* experiments/ contains the training, testing and visualize scripts.

* `config.py` is the configuration file to set the training and testing parameters.

* `train_metric.py` is the main function to start training the model.

* `test_proto.py` and `test_metric_knn.py` are the main functions to start testing model.

* `visualization.py` is the script to visualize testing result.
 

### Prerequisites

* Python 3.6+
* PyTorch 1.9.0
* pytorch-metric-learning
* segmentation_models_pytorch
* torchvision 0.2.1+
* pycocotools
* sacred 0.7.5
* tqdm 4.32.2


### Installation

1. Download the source code
2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```

### Preparation

1. Prepare Pascal-5i dataset

   * Download VOC2012 from [officail website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) and put them under ./data/Pascal/

   * Download SegmentationClassAug, SegmentationObjectAug, ScribbleAugAuto from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp) and put them under ./data/Pascal/VOCdevkit/VOC2012/
   
   * Download Segmentation from [here](https://drive.google.com/drive/folders/1N00R9m9qe2rKZChZ8N7Hib_HR2HGtXHp) and use it to replace VOCdevkit/VOC2012/ImageSets/Segmentation.

2. Prepare MSCOCO dataset
   * Download COCO 2014 form [officail website](https://cocodataset.org/#download) and put them under ./data/COCO/

3. Prepare pretrain model
   * Download pretrained model [here](https://drive.google.com/file/d/1S2CJNmkLMTW1qQNfIc_JJaR88pefqXqd/view?usp=sharing) and put them under ./


<!-- USAGE EXAMPLES -->
## Usage

1. Train the model
   ```
   sh experiments/train.sh
   ``` 

2. Test the model
    ```
    sh experiments/test.sh
    ```

3. Visualize
   ```
   sh experiments/vis.sh
   ```

<!-- REFERENCES -->
## References

* [PANet](https://github.com/xumengen/PANet)
* [pytorch](https://pytorch.org/)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
