"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import os.path

import numpy as np
from PIL import Image

from .generator import Generator
from ..utils.image import read_image_bgr

#visdataset_classes = {
    #'Pedestrian': 0,
    #'Person': 1,
    #'Car': 2,
    #'Van': 3,
    #'Bus': 4,
    #'Truck': 5,
    #'Motor': 6,
    #'Bicycle': 7,
    #'Awning-tricycle':8,
    #'Tricycle':9 
#}

visdataset_classes = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10':10,
    '11':11
}


class visdatasetGenerator(Generator):
    """ Generate data for a KITTI dataset.

    See http://www.cvlibs.net/datasets/kitti/ for more information.
    """

    def __init__(
        self,
        base_dir,
        subset='train',
        **kwargs
    ):
        """ Initialize a visdataset data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            subset: The subset to generate data for (defaults to 'train').
        """
        self.base_dir = base_dir

        label_dir = os.path.join(self.base_dir, subset, 'annotations')
        image_dir = os.path.join(self.base_dir, subset, 'images')
        #label_dir = 'C:/Users/14590/Desktop/Victor/dataset/vis/test/annotations'
        #image_dir = 'C:/Users/14590/Desktop/Victor/dataset/vis/test/images'


        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Pe rson_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """

        self.labels = {}
        self.classes = visdataset_classes
        for name, label in self.classes.items():
            self.labels[label] = name

        self.image_data = dict()
        self.images = []
        for i, fn in enumerate(os.listdir(label_dir)):
            label_fp = os.path.join(label_dir, fn)
            image_fp = os.path.join(image_dir, fn.replace('.txt', '.jpg'))

            self.images.append(image_fp)

            fieldnames = ['left','top','width','height','score','type','truncated', 'occluded']
            if os.path.isfile(label_fp):
                #print(label_fp)
                with open(label_fp, 'r') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=',', fieldnames=fieldnames)
                    boxes = []
                    for line, row in enumerate(reader):
                        label = row['type']#字符
                        cls_id = visdataset_classes[label]#1对应1
                        #cls_id=clsid

                        annotation = {'cls_id': cls_id, 'x1': row['left'], 'x2': int(row['left'])+int(row['width']), 'y2': int(row['top'])+int(row['height']), 'y1': row['top']}
                        boxes.append(annotation)

                    self.image_data[i] = boxes

        super(visdatasetGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.images)

    def num_classes(self):
        """ Number of classe s in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError()

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.images[image_index])
        return float(image.width) / float(image.height)

    def image_path(self, image_index):
        """ Get the path to an image.
        """
        return self.images[image_index]

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        image_data = self.image_data[image_index]
        annotations = {'labels': np.empty((len(image_data),)), 'bboxes': np.empty((len(image_data), 4))}

        for idx, ann in enumerate(image_data):
            annotations['bboxes'][idx, 0] = float(ann['x1'])
            annotations['bboxes'][idx, 1] = float(ann['y1'])
            annotations['bboxes'][idx, 2] = float(ann['x2'])
            annotations['bboxes'][idx, 3] = float(ann['y2'])
            annotations['labels'][idx] = int(ann['cls_id'])

        return annotations



    def cal_instances(self):
        """ Load annotations for an image_index.
        """
        ll=len(visdataset_classes)
        contents=[0 for index in range(ll)]
        for key in self.image_data.keys():
            azz=len(self.image_data.get(key))
            for l in range(0,azz):
                if self.image_data.get(key)[l]['cls_id'] ==0:
                    contents[0]=contents[0]+1
                elif self.image_data.get(key)[l]['cls_id'] ==1:
                    contents[1]=contents[1]+1
                elif self.image_data.get(key)[l]['cls_id'] ==2:
                    contents[2]=contents[2]+1
                elif self.image_data.get(key)[l]['cls_id'] ==3:
                    contents[3]=contents[3]+1
                elif self.image_data.get(key)[l]['cls_id'] ==4:
                    contents[4]=contents[4]+1
                elif self.image_data.get(key)[l]['cls_id'] ==5:
                    contents[5]=contents[5]+1
                elif self.image_data.get(key)[l]['cls_id'] ==6:
                    contents[6]=contents[6]+1
                elif self.image_data.get(key)[l]['cls_id'] ==7:
                    contents[7]=contents[7]+1
                elif self.image_data.get(key)[l]['cls_id'] ==8:
                    contents[8]=contents[8]+1
                elif self.image_data.get(key)[l]['cls_id'] ==9:
                    contents[9]=contents[9]+1
                elif self.image_data.get(key)[l]['cls_id'] ==10:
                    contents[10]=contents[10]+1
                elif self.image_data.get(key)[l]['cls_id'] ==11:
                    contents[11]=contents[11]+1 

        print('class 1 has '+ str(contents[0]) + ' instances')
        print('class 2 has '+ str(contents[1]) + ' instances')
        print('class 3 has '+ str(contents[2]) + ' instances')
        print('class 4 has '+ str(contents[3]) + ' instances')
        print('class 5 has '+ str(contents[4]) + ' instances')
        print('class 6 has '+ str(contents[5]) + ' instances')
        print('class 7 has '+ str(contents[6]) + ' instances')
        print('class 8 has '+ str(contents[7]) + ' instances')
        print('class 9 has '+ str(contents[8]) + ' instances')
        print('class 10 has '+ str(contents[9]) + ' instances')
        print('class 11 has '+ str(contents[10]) + ' instances')
        print('class 12 has '+ str(contents[11]) + ' instances')
        return contents


