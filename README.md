# CB-retinanet
Class balanced retinanet with Visdrone dataset. Followed by [Keras-retinanet](https://github.com/fizyr/keras-retinanet.git), we change the focal loss to class-balanced focal loss(https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) under retinanet and add VisDrone-2019 dataset as one of the trainingsets.

## Installation

1) Clone this repository.
2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
3) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Training on VisDrone-019 dataset
```
# Running directly from the repository:
keras_retinanet/bin/train.py visdataset /path/to/VisDrone/dataset
```

## More information please reference to [keras-retinanet](https://github.com/fizyr/keras-retinanet)





## Acknowledgement
This code is extended from the following repositories.
- [keras-retinanet](https://github.com/fizyr/keras-retinanet)
Thank the authors for releasing their codes. Please also consider citing their works.


