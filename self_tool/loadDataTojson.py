import pandas as pd
import numpy as np
import os
import json

# 读取Excel文件
# file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0604.xlsx'
# data_path = '/home/local/ljl/data/yolo_data/data_set/train/'
# output_path = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0604.json'
# isVAL = False

# 生成val数据
file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0523.xlsx'
data_path = '/home/local/ljl/data/yolo_data/data_set/val/'
output_path = '/home/local/ljl/data/yolo_data/data_set/annotations/gxd_road_yolo_algo_0523.json'
isVAL = True

def processContourTobbox(segment_coords):
    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')
    # 遍历每个坐标点，更新边界值
    for coord in segment_coords:
        x, y = coord
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    # 计算宽度和高度
    width = x_max - x_min
    height = y_max - y_min

    bbox_format = [x_min, y_min, width, height]

    return bbox_format


def transform_bbox(bbox, scale, padding):
    """
    转换边界框坐标
    bbox: [x_min, y_min, width, height]
    orig_size: (width, height) 原始图像尺寸
    scale: 缩放比例
    padding: (pad_left, pad_top) 填充偏移量
    """
    x_min, y_min, w, h = bbox
    pad_left, pad_top = padding
    
    # 应用缩放
    x_min_scaled = x_min * scale
    y_min_scaled = y_min * scale
    w_scaled = w * scale
    h_scaled = h * scale
    
    # 应用填充偏移
    x_min_final = x_min_scaled + pad_left
    y_min_final = y_min_scaled + pad_top
    
    return [int(x_min_final), int(y_min_final), int(w_scaled), int(h_scaled)]


def generateJson(file_path):
    idx = 1
    res_map = {}
    df = pd.read_excel(file_path)
    for index, row in df.iterrows():
        imageFilename = row['image_id'] + '.jpg'
        imagePath = data_path + imageFilename
        
        if not os.path.exists(imagePath):
            continue

        print('imagePath:', imagePath)
        if imageFilename not in res_map:
            res_map[imageFilename] = {}
            res_map[imageFilename]['bboxes'] = []
            res_map[imageFilename]['labels'] = []
            res_map[imageFilename]['segmentation'] = []
            res_map[imageFilename]['filename'] = imageFilename
            res_map[imageFilename]['seg_filename'] = imageFilename
            res_map[imageFilename]['id'] = idx
            res_map[imageFilename]['width'] = 640
            res_map[imageFilename]['height'] = 640

            idx += 1

        segment_coords = json.loads(row['contour'])
        bbox = processContourTobbox(segment_coords)

        new_bbox = transform_bbox(bbox, 0.5, (80, 0))

        res_map[imageFilename]['bboxes'].append(new_bbox)
        res_map[imageFilename]['labels'].append(1)
        res_map[imageFilename]['segmentation'].append(segment_coords)

        if isVAL and idx > 10000:
            break
    res_list = list(res_map.values())
    result = json.dumps(res_list, ensure_ascii=False)
    with open(output_path, 'w') as f:
        f.write(result)

def main():
    generateJson(file_path)        

# 示例使用
if __name__ == "__main__":
    main()