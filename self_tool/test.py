# 定义段坐标列表
segment_coords = [[172, 541], [168, 545], [174, 675], [176, 679], [190, 679], [185, 550], [181, 541]]

# 初始化变量
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

# 输出两种常见的边界框格式
bbox_format_1 = [x_min, y_min, width, height]
bbox_format_2 = [x_min, y_min, x_max, y_max]

print("Bounding box format 1 (x_min, y_min, width, height):", bbox_format_1)
print("Bounding box format 2 (x_min, y_min, x_max, y_max):", bbox_format_2)
