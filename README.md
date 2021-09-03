<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Apply Deep Metric Learning to Few-Shot Image Segmentation</h3>

  <p align="center">
    Individual Project
    <br />
    <a href="https://github.com/xumengen/FSGM"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/xumengen/FSGM">View Demo</a>
    ·
    <a href="https://github.com/xumengen/FSGM/issues">Report Bug</a>
    ·
    <a href="https://github.com/xumengen/FSGM/issues">Request Feature</a>

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
    <li><a href="#folder-structure">Folder Structure</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#preparation">Preparation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project is code implementation of my final project report in KCL. In this repo, deep metric learning is applied to few shot segmentation. According to the information provided by the mask, the distance between feature vectors of the same category in the high-dimensional space is as small as possible, and the distance between feature vectors of different categories is close or far away.


### Built With

* [pytorch](https://pytorch.org/)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)


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


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

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


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/xumengen/FSGM/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Project Link: [https://github.com/xumengen/FSGM](https://github.com/xumengen/FSGM)

<!-- ACKNOWLEDGEMENTS -->
## References

* [PANet](https://github.com/xumengen/PANet)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)