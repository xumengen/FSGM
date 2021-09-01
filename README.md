<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Apply Deep Metric Learning to Few-Shot Image Segmentation</h3>

  <p align="center">
    Individual Project

</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
In this repo, deep metric learning is applied to few shot segmentation. According to the information provided by the mask, the distance between feature vectors of the same category in the high-dimensional space is as small as possible, and the distance between feature vectors of different categories is close or far away.


### Built With

* [pytorch](https://pytorch.org/)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models](https://github.com/qubvel/segmentation_models)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* Python 3.6+
* PyTorch 1.9.0
* pytorch-metric-learning
* segmentation_models
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
    ```sh
    sh experiments/train.sh
    ```

2. Test the model
    ```sh
    sh experiments/test.sh
    ```

3. Visualize
   ``` sh
   sh experiments/vis.sh
   ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- ACKNOWLEDGEMENTS -->
## References

* [PANet](https://github.com/xumengen/PANet)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models](https://github.com/qubvel/segmentation_models)