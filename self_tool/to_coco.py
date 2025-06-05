
import datetime
import math
import json
import os
import re
import cv2
import fnmatch
from PIL import Image
import numpy as np


if True:
    input_json = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0523.json'
    output_json = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0523_coco.json'
if False:
    input_json = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0604.json'
    output_json = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0604_coco.json'


if os.path.isfile(output_json):
    print(f'{output_json} 已存在.')
    exit()


INFO = {
    "description": "Board Completeness Dataset",
    "url": "https://yuque.antfin-inc.com/amap-poi",
    "version": "1.0.0",
    "year": 2022,
    "contributor": "xuhaoyu.xhy",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
}]

CATEGORIES = [{
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
},
            {
        'id': 2,
        'name': 'board',
        'supercategory': 'board',
}]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def create_image_info(image_id,
                      file_name,
                      image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1,
                      coco_url="",
                      flickr_url=""):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id,
                           image_id,
                           category_id,
                           area,
                           image_size=None,
                           bounding_box=None,
                           segmentation=None,
                           is_crowd=0):

    print ('area:{}, bounding_box:{}, segmentation:{}, category_id:{}'.format(area, bounding_box, segmentation, category_id))
    if area < 1:
        return None
    if bounding_box is None:
        return None
    if segmentation is None:
        return None

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area,
        "bbox": bounding_box,
        "segmentation": segmentation,
        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info


def check_polygon(polygon, image_width, image_height):
    _, idx = np.unique(polygon, axis=0, return_index=True)
    polygon = polygon[np.sort(idx)]

    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)

    if len(polygon) < 3:
        return None

    pts_area = cv2.contourArea(polygon)
    if pts_area < 0:
        return None
    return polygon

def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # image_id = 1
    annotation_id = 1

    with open(input_json, 'r') as file:
        data = json.load(file)

    for sample in data:
        image_id = sample['id']
        bboxes = sample['bboxes']
        labels = sample['labels']
        segmentation = sample['segmentation']
        filename = sample['filename']
        width = sample['width']
        height = sample['height']

        is_error_img = False

        image_info = create_image_info(image_id,
                                       filename,
                                       (width, height))

        ann_info_list = list()
        pre_annotation_id = annotation_id
        for bbox, points, label in zip(bboxes, segmentation, labels):
            polygon = np.array(points).reshape(-1, 2).astype(np.float32)
            area = cv2.contourArea(polygon)

            # ljl tmp 因为不考虑segment 所以area设置为一个满足要求的值
            area = 10
            if area < 8:
                is_error_img = True

            segmentation = [polygon.flatten().tolist()]
            is_crowd = 0

            category_id = CATEGORIES[label - 1]['id']

            ann_info = create_annotation_info(annotation_id, image_id, category_id, area, (width, height), bbox,
                                              segmentation, is_crowd)

            ann_info_list.append(ann_info)

            annotation_id = annotation_id + 1

        if is_error_img == False:
            coco_output["images"].append(image_info)
            coco_output["annotations"].extend(ann_info_list)
            # image_id = image_id + 1
        else:
            annotation_id = pre_annotation_id


    with open(output_json, 'w') as output_json_file:
        output_json_file.write(json.dumps(coco_output, ensure_ascii=False, indent=2))

    print(f'save to {output_json}.')


if __name__ == '__main__':
    main()