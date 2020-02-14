# mask_galaxy

The classification of galaxies based on their morphology is instrumental to the understanding of galaxy formation and evolution. This, in addition to the ever-growing digital astronomical datasets, has motivated the application of advanced computer vision techniques, such as Deep Learning. But these proposals have not allowed us to have a single pipeline that replicates detection, segmentation and morphological classification of galaxies made by experts. The process has been performed either visually or through relied on semi-automated software, mainly SExtractor. We present the implementation of a automatic machine learning pipeline for detection, segmentation and morphological classification of galaxies. Model based on Deep Learning architecture: Mask R-CNN. This state-of-the-art model of Instance Segmentation also performs image segmentation at the pixel level, which is a recurrent need in the astronomical community. We achieve a Mean Average Precision (mAP) of 0.93 in morphological classification of Spiral or Elliptical galaxies.

## Resources needed for the model

- [Necessary python libraries](http://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/requirements.txt)

- [Dataset](http://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/requirements.txt)

- [Weights of the network](http://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/galaxia_all_1.h5)

- [Training dataset configuration file](https://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/train_via_region_data.json)

- [Validation dataset configuration file](https://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/val_via_region_data.json)

- [Galaxy zoo catalog](http://datascience-userena.s3.amazonaws.com/mask_galaxy-morphological_segmentation_of_galaxies/zoo2MainSpecz_simpleLabels.csv)

### Configuration steps

- Install all libraries in the requirements.txt file. In their respective versions.

- Download the configuration files of the training and validation dataset. When loading them in the respective folders leave the following name for both files: via_region_data.json

- Unzip the imageset folder of the dataset. Duplicate it and incorporate the respective configuration files.

- Download the weights and leave them in the "models" folder.


