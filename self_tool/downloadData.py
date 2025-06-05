import pandas as pd
import requests

# 读取Excel文件
# file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0523.xlsx'
file_path = '/home/local/ljl/data/yolo_data/gxd_road_yolo_algo_0604.xlsx'
df = pd.read_excel(file_path)

# 遍历每一行，下载图片
for index, row in df.iterrows():
    image_id = row['image_id']
    url = row['url']
    
    # 下载图片
    response = requests.get(url)
    saveDataPath = "/home/local/ljl/data/yolo_data/data_set/images/"
    if response.status_code == 200:
        with open(f'{saveDataPath}{image_id}.jpg', 'wb') as f:
            f.write(response.content)
        print(f'Successfully downloaded {image_id}')
    else:
        print(f'Failed to download {image_id}, status code: {response.status_code}')

print('Download process completed.')