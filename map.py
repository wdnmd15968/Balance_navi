from PIL import Image
import numpy as np

# 定义图像的宽度和高度
width = 64
height = 64

# 生成随机的灰阶像素数据
pixels = np.random.randint(1, 3, (height, width), dtype=np.uint8)

# 创建一个新的图像对象
img = Image.fromarray(pixels, 'L')

# 保存图像到文件
img.save('heightmap.png')