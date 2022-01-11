# Laser-Spot-Detector
## Introduction

This is an open source laser spot detector based on boundary point set, which is suitable for complex indoor scenes.

![Detection Results](Detection.svg)

## Dependencies

- [OpenCV](https://docs.opencv.org/4.x/index.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [scikit-learn](https://scikit-learn.org/stable/)

## File

The structure of files is as follows:
```
  ├── Checkpoint
  │   ├── Ensemble.model
  │   ├── resnet50.pth
  ├── Config
  │   ├── config.json
  ├── Dataset
  ├── Demo
  │   ├── 1.jpg
  │   ├── ···
  │   ├── 8.jpg
  ├── TestSet
  │   ├── GroundTruth
  │   │   ├── xxx.json
  │   ├── TestData
  │   │   ├── xxx.jpg(or other format)
  ├── TrainingSet
  │   ├── negative
  │   │   ├── xxx.jpg(or other format)
  │   ├── positive
  │   │   ├── xxx.jpg(or other format)
  │   ├── positive_augmented
  │   │   ├── xxx.jpg(or other format)
  │   ├── negative.csv
  │   ├── positive_augmented.csv
  ├── Demo.py
  ├── Detection.svg
  ├── Laser_Detector.py
  ├── README.md
  ├── requirements.txt
  ├── test.py
  ├── train.py
```

## Demo

We provide a demo script to detect some images with the laser spot.

```
python Demo.py --config ${CONFIG_FILE} -i ${INPUT_PATH} -o ${OUTPUT_PATH}
```

Arguments:

- `CONFIG_FILE`: The path of config file.
- `INPUT_PATH`: The path of the images to be detected with the laser spots, which can be the path of the folder or the specified image.
- `OUTPUT_PATH`: The output path of the detected images.

For example:

```shell
python Demo.py --config Config/config.json -i Demo/ -o Demo/
```

## Test

We can run the script to test the precision and recall in the test set.

```shell
python Demo.py --config ${CONFIG_FILE} -i ${INPUT_PATH} [-o ${OUTPUT_PATH}]
```

Arguments:

- `CONFIG_FILE`: The path of config file.

- `INPUT_PATH`: The path of the test set folder. The structure of test set folder should be as follows.

  ```
  ├── TestSet
  │   ├── GroundTruth
  │   │   ├── xxx.json
  │   │   ├── ···
  │   ├── TestData
  │   │   ├── xxx.jpg(or other format)
  │   │   ├── ···
  ```

​	Where `GroundTruth` contains the laser spot exact position in the image of the same name.

- `OUTPUT_PATH`: Optional. The output path of the detected images. If it is not specified, only the precision and recall will be calculated and the detected images won't be stored.

For example:

```shell
python test.py --config Config/config.json -i TestSet/
```

## Train

You can run the `train.py` to train the model in your own dataset.

```shell
python train.py --config ${CONFIG_FILE} -p ${POSIIVE_FILE} -n ${NEGATIVE_FILE} -o ${OUTPUT_PATH} --max_iter ${NUM_ITER} [-e]
```

Arguments:

- `CONFIG_FILE`: The path of config file.
- `POSITIVE_PATH`: The path of the positive images or the features of positive images. 
- `NEGATIVE_PATH`: The path of the negative images or the features of negative images. 
- `OUTPUT_PATH`: The output path of the trained model.
- `NUM_ITER`: The number of training iterations.
- `-e`: Options. If it is specified, `POSITIVE_PATH` and `NEGATIVE_PATH` are the path of the images and the features are extracted from the images. Otherwise, `POSITIVE_PATH` and `NEGATIVE_PATH` are the path of the features.

For example:

```shell
python train.py --config Config/config.json -p TrainingSet/positive_augmented -n TrainingSet/negative -o Checkpoint/Ensemble.model --max_iter 10000
```

## Config

The configuration file parameters are as follows:

- `classifier_param`: The path of classifier model.
- `network_param_pth`: The path of pretrained network model.
- `range_of_filter`: The range of laser spots in HSV. If the list contains more than one element, it means the union of ranges. Such as `[{'low': [150, 20, 220], 'up': [180, 255, 255]},{'low': [0, 20, 220], 'up': [10, 255, 255]}]`.
- `grad_thresh`: Threshold to binarize gradient of image.
- `args_in_CDBPS`: It contains the parameters in CDBPS. `minVar` refers to threshold of radius variance $\eta$, `minRadius` refers to the minimum radius $R_{min}$ and `maxRadius` refers to the maximum radius $R_{max}$.
- `output_is_circle`: If it is `True`, the output form of the laser spot area is circular. Or the form is a rectangle.
- `structure_size`: The size of structuring element for closing operation.

```python
detector = Detector(
    classifier_param='Checkpoint/Ensemble.model',
    network_param_pth='Checkpoint/resnet50.pth',
    range_of_filter=[{'low': [150, 20, 220], 'up': [180, 255, 255]}],
    grad_thresh=50,
    args_in_CDBPS={'minVar': 5, 'minRadius': 2, 'maxRadius': 15},
    output_is_circle=False,
    structure_size=7
)
```