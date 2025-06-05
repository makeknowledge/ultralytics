from PIL import Image
import cv2
import pandas as pd
import os
import shutil

def move_file(src_path, dest_dir):
    """
    移动文件到指定目录
    
    :param src_path: 要移动的文件的完整路径
    :param dest_dir: 目标目录的路径
    """
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 提取文件名
    file_name = os.path.basename(src_path)
    
    # 构建目标文件的完整路径
    dest_path = os.path.join(dest_dir, file_name)
    
    # 移动文件
    shutil.move(src_path, dest_path)
    print(f"File moved from {src_path} to {dest_path}")


def main():
    file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0523.xlsx'
    df = pd.read_excel(file_path)
    data_path = '/home/local/ljl/data/yolo_data/data_set/images'

    # 遍历每一行，下载图片
    for index, row in df.iterrows():
        image_filename = row['image_id'] + '.jpg'
        
        image_file = os.path.join(data_path, image_filename)
        if not os.path.exists(image_file):
            continue

        print('file:', image_file)
        image = Image.open(image_file)

        # 原始尺寸 (W, H)
        orig_w, orig_h = image.size

        print('file:{}, orig_w:{}, orig_h:{}'.format(image_file, orig_w, orig_h))
        #将文件移动到另一个文件夹
        src_path = '/home/local/ljl/data/yolo_data/data_set/images/' + image_filename
        dest_dir = '/home/local/ljl/data/yolo_data/data_set/val/'
        move_file(src_path, dest_dir)

if __name__ == '__main__':
    main()
    # test()