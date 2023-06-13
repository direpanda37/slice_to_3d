## references
  [*Ronneberger et al, 2015, U-Net: Convolutional Networks for Biomedical Image Segmentation*](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
  
  [*Machireddy et al, 2021, Robust Segmentation of Cellular Ultrastructure on Sparsely Labeled 3D Electron Microscopy Images using Deep Learning*](https://www.biorxiv.org/content/10.1101/2021.05.27.446019v1.full)

## objectives
- **Proof of Concept**
- Streamline the use of volumetric electron microscopy (vEM) cell images to create a 3D model for the purpose of gaining *"deeper understanding of the cellular and subcellular organization of tumor cells and their interactions with the tumor microenvironment (to) shed light on how cancer evolves and guide effective therapy choices."*
- Provide models and tools for researchers to tailor to meet their needs

## workflow overview

*Implemented in main_create_3d_volume.ipynb with support code in model.py and data.py*

Using electron microscopy (vEM) cell images to create a 3D model is as follows:
- Take a stack of vEM slices of tissue and 
  - pre-process the images as needed and scale them to 512x512
- If training the model
  - optionally apply data augmentation to increase the training dataset and push the images through the the training process
  - optionally, load a pretrained model and use additional training to fine-tune
  - train the model, saving checkpoints periodically and reporting progress as the model is training
  - save the final model parameters/weights
- If segmenting images with an existing model
  - create the model and load pretrained weights into the model
  - convert each image into a segmented image using a pretrained UNet model
  - save the segmented images as PNG files
- Load the stack of images as a 3D [NumPy](https://numpy.org/doc/stable/) array using [imageio.imread()](https://imageio.readthedocs.io/en/v2.16.1/_autosummary/imageio.imread.html).
- Use the [marching cubes algorithm](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) from the scikit-image submodule `skimage.measure` to conver

## UNet Model
![UNet Architecture](img/u-net-architecture.png)
This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function makes sure that mask pixels are in [0, 1] range.

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow (used 2.12), Keras >= 1.0 for building a model, training/loading weights, and segmenting images
* Scikit.measure, STL for constructing 3D model

Also, this code should be compatible with Python versions 2.7-3.5.

### follow the main_create_3d_volume notebook

You will see the predicted results of test image in data/membrane/test and a 3d model in stl

## additional resources

[CREMI: MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images](https://cremi.org/) Ancient challenge (2016), but has datasets may be worth exploring: 1250  × 1250 × 125  fly brain images. Each volume has neuron and synapse labeling and annotations for pre- and post-synaptic partners.

[MitoEM Challenge: Large-scale 3D Mitochondria Instance Segmentation](https://mitoem.grand-challenge.org/) A more recent challenge (2021).More datasets that are worth exploring: 1000x4096x4096 mitochondria image volumes are acquired from a rat (Mito-R) and a human (Mito-H) tissue

[Din & Yu in Nature: Training a deep learning model for single-cell segmentation without manual annotation](https://www.nature.com/articles/s41598-021-03299-4#code-availability) "Generating training data is expensive and a major hindrance in the wider adoption of machine learning based methods for cell segmentation. Here we present an alternative strategy that trains CNNs without any human-labeled data. We show that our method is able to produce accurate segmentation models, and is applicable to both fluorescence and bright-field images, and requires little to no prior knowledge of the signal characteristics."


