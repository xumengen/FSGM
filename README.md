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
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
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

1. Clone the repo
   ```sh
   git clone https://github.com/xumengen/FSGM
   ```
2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```


<!-- USAGE EXAMPLES -->
## Usage

1. Download data from [here]() and put them under FSGM/ .

2. Download pretrained weights from [here]() and put them under FSGM/pretrained_model/ .

3. Train the model
    ```sh
    sh experiments/train.sh
    ```

4. Test the model
    ```sh
    sh experiments/test.sh
    ```

5. Visualize
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
## Acknowledgements

* [PANet](https://github.com/xumengen/PANet)
* [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* [segmentation_models](https://github.com/qubvel/segmentation_models)