# 这个文件用来可视化coco json
from os.path import join
from multiprocessing.pool import ThreadPool
from icecream import ic
import os
import cv2
import numpy as np
import json

DRAW_GT = True
SAVE_IMAGE = True

NUM_WORKERS = 15

if True:
    image_root = '/home/local/ljl/data/yolo_data/data_set/val/'
    json_path = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0523_coco.json'

save_root = '/home/local/ljl/data/yolo_data/data_set/vis/'
os.makedirs(save_root, exist_ok=True)

with open(json_path, 'r') as file:
    data = json.load(file)
print(f'load "{json_path}" ok.')

category_list = [None]
category_list += [category['name'] for category in data['categories']]

images = {}

thresh = 0.1

for image in data['images']:
    file_name = image['file_name']
    image_id = image['id']
    images[image_id] = {}
    images[image_id]['file_name'] = file_name
    images[image_id]['bbox'] = []
    images[image_id]['segmentation'] = []
    images[image_id]['category_id'] = []

for annotation in data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    segmentation = annotation['segmentation'][0]
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    polylines = list(zip(xs, ys))
    images[image_id]['bbox'].append(bbox)
    images[image_id]['segmentation'].append(polylines)
    images[image_id]['category_id'].append(annotation['category_id'])

annotations = data['annotations']
ic(len(annotations))  # 删除mask内部gt前的

cur = 0
total = 0
def worker(image_id):
    global cur, total
    sample = images[image_id]
    file_name = sample['file_name']

    file_path = join(image_root, file_name)
    assert os.path.isfile(file_path), f'{file_path} not exist.'
    image = cv2.imread(file_path)

    bboxes = sample['bbox']
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    save_path = join(save_root, file_name)

    segmentations = sample['segmentation']
    category_ids = sample['category_id']

    if SAVE_IMAGE:
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'{cur}/{total} save to "{save_path}".')
    else:
        print(f'{cur}/{total} image not saved.')

    cur += 1

jobs = []
for i, image_id in enumerate(images):
    jobs.append(image_id)

total = len(jobs)
if NUM_WORKERS:
    pool = ThreadPool(processes=5)
    job_out = pool.map(worker, jobs)
    pool.close()
    pool.join()
else:
    for job in jobs:
        worker(job)