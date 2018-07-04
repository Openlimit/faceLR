import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = []
y = []
z = []
with open('stdhead.obj',mode='r') as f:
    for i in range(2455):
        line = f.readline()
        strs = line.split(' ')
        x.append(float(strs[1]))
        y.append(float(strs[2]))
        z.append(float(strs[3]))


ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x, y, z, c='y')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
