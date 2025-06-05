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

def letterbox_resize(image, target_size=(640, 640)):
    """
    保持宽高比的 resize，返回处理后的图像和变换参数
    """
    # 原始尺寸 (W, H)
    orig_w, orig_h = image.size
    
    # 目标尺寸 (W, H)
    tgt_w, tgt_h = target_size
    
    # 计算缩放比例 (保持宽高比)
    scale = min(tgt_w / orig_w, tgt_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # 执行缩放
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    
    # 创建新图像并粘贴缩放后的图像
    new_image = Image.new('RGB', (tgt_w, tgt_h), (114, 114, 114))  # 灰色填充
    left = (tgt_w - new_w) // 2
    top = (tgt_h - new_h) // 2
    new_image.paste(resized, (left, top))
    
    # 返回处理后的图像和变换参数
    return new_image, scale, (left, top)

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
    
    return [x_min_final, y_min_final, w_scaled, h_scaled]



def main():
    file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0604.xlsx'
    df = pd.read_excel(file_path)
    data_path = '/home/local/ljl/data/yolo_data/data_set/images/'

    # 遍历每一行，下载图片
    for index, row in df.iterrows():
        image_filename = row['image_id'] + '.jpg'
        
        image_file = os.path.join(data_path, image_filename)
        if not os.path.exists(image_file):
            continue

        image = Image.open(image_file)

        # 原始尺寸 (W, H)
        orig_w, orig_h = image.size
        if orig_w == 640 and orig_h == 640:
            continue

        if orig_w != 960 or orig_h != 1280:
            print('file:{}, orig_w:{}, orig_h:{}'.format(image_file, orig_w, orig_h))
            #将文件移动到另一个文件夹
            src_path = '/home/local/ljl/data/yolo_data/data_set/images/' + image_filename
            dest_dir = '/home/local/ljl/data/yolo_data/data_set/images_error_size/'
            move_file(src_path, dest_dir)
            continue

        target_size = (640, 640)
        new_image, scale, padding = letterbox_resize(image, target_size)
        new_image.save(image_file)
        print('file:{}, scale:{}, padding:{}'.format(image_file, scale, padding))

if __name__ == '__main__':
    main()
    # test()





def test():
    ImgPath = "/home/local/ljl/data/yolo_data/data_set/tmp_image/02a93144f27adc8d4f813708a7e3f7ba_d4bfb675af06b744331400053beb4c0d.jpg"
    image = Image.open(ImgPath)
    target_size = (640, 640)
    new_image, scale, padding = letterbox_resize(image, target_size)

    segment = [[549,384],[344,416],[351,501],[552,479]]

    bbox = processContourTobbox(segment)

    new_bbox = transform_bbox(bbox, scale, padding)

    print('new_bbox:',new_bbox)

    x1, y1, w, h = new_bbox
    x2 = x1 + w
    y2 = y1 + h

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


    new_image.save('/home/local/ljl/data/yolo_data/data_set/tmp_image/resize.jpg')
    print('scale:{}, padding:{}'.format(scale, padding))

    img = cv2.imread('/home/local/ljl/data/yolo_data/data_set/tmp_image/resize.jpg')
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('/home/local/ljl/data/yolo_data/data_set/tmp_image/resize_bbox.jpg', img)