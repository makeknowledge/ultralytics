from misc_utils import load_json
from os.path import join
from multiprocessing.pool import ThreadPool
import os
import cv2
import numpy as np

DRAW_GT = True
SAVE_IMAGE = True

NUM_WORKERS = 15

if True:
    image_root = '/workspace/dataset/blur_filled_data/images'
    json_path = '/workspace/dataset/blur_filled_data/annotations/ele_2w_cohesion.json'

save_root = './vis'
os.makedirs(save_root, exist_ok=True)

data = load_json(json_path)
print(f'load "{json_path}" ok.')

cur = 0
total = 0
def worker(idx):
    global cur, total
    sample = data[idx]
    file_name = sample['filename']

    file_path = join(image_root, file_name)
    assert os.path.isfile(file_path), f'{file_path} not exist.'
    image = cv2.imread(file_path)

    bboxes = sample['bboxes']
    for bbox in bboxes:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    save_path = join(save_root, file_name)

    segmentations = sample['segmentation']
    labels = sample['labels']

    gts = np.zeros_like(image)
    for segmentation, label in zip(segmentations, labels):
        # 画真值
        if DRAW_GT:
            poly = np.array(segmentation, np.int32).reshape(-1, 2)
            cv2.fillPoly(gts, [poly], color=[0, 255, 255])
            cv2.polylines(image, [poly], True, (0, 255, 255), 3)
            xmin = min(poly[:,0])
            ymin = min(poly[:,1])
            xmax = max(poly[:,0])
            ymax = max(poly[:,1])
            category_name = str(label)
            cv2.putText(image, category_name, (int(xmin+xmax)//2 - 50,int(ymax+ymin)//2 ), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    if DRAW_GT:
        image = cv2.addWeighted(image, 1., gts, 0.2, 0)

    if SAVE_IMAGE:
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'{cur}/{total} save to "{save_path}".')
    else:
        print(f'{cur}/{total} image not saved.')

    cur += 1

jobs = []
for i, sample in enumerate(data):
    jobs.append(i)

total = len(jobs)
if NUM_WORKERS:
    pool = ThreadPool(processes=5)
    job_out = pool.map(worker, jobs)
    pool.close()
    pool.join()
else:
    for job in jobs:
        worker(job)