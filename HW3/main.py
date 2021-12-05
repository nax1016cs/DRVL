import os
from pycococreatortools import pycococreatortools
import numpy as np
import json
import re
from PIL import Image

INFO = {
    "description": "Nuclei Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "ming",
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'nuclei ',
        'supercategory': 'shape',
    }
]
IMAGE_DIR = "./dataset/train/"
# filter for jpeg images

coco_output = {
    "info" : INFO,
    "license" : LICENSES,
    "categories" : CATEGORIES,
    "images" : [],
    "annotation" : []
}
image_id = 1
segmentation_id = 1
for dir_ in os.listdir(IMAGE_DIR):
    img_name = dir_ + ".png"
    image_filename = os.path.join(IMAGE_DIR, dir_, "images", img_name)
    mask_path = os.path.join(IMAGE_DIR, dir_, "masks")
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(
        image_id, os.path.basename(image_filename), image.size)
    coco_output["images"].append(image_info)

    # filter for associated png annotations
    for maskid in os.listdir(mask_path):
        mask_filename = os.path.join(mask_path, maskid)
        category_info = {'id': 1, 'is_crowd': 0}
        binary_mask = np.asarray(Image.open(mask_filename)
            .convert('1')).astype(np.uint8)
        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id, image_id, category_info, binary_mask,
            image.size, tolerance=2)
        coco_output["annotation"].append(annotation_info)
        segmentation_id += 1
    image_id += 1
    print("finish ", image_filename)
with open ('./nuclei.json', 'w') as outputfile:
    json.dump(coco_output, outputfile, indent=4)