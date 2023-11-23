import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 14   # 设置字体大小为14
rcParams['font.family'] = 'serif'   # 设置字体为衬线字体

methods = ['AADB', 'CLIP', 'GATx3_GATP', 'MUSIQ', 'NIMA_mobile', 'NIMA_VGG16',
           'NIMA_InceptionV2', 'TANet', 'VILA', 'ReLIC', 'Ours']
params = [11.18, 151.28, 49.42, 78.59, 3.40,
          18.70, 11.26, 13.88, 383, 37.77, 200]
ACC = [77.33, 81.60, 82.15, 81.50, 80.36, 80.60,
       81.51, 80.64, 82.88, 82.35, 84.89]
SRCC = [0.558, 0.731, 0.762, 0.726, 0.510, 0.592,
        0.612, 0.758, 0.774, 0.748, 0.836]
PLCC = [0.580, 0.741, 0.764, 0.738, 0.518, 0.610,
        0.636, 0.765, 0.774, 0.760, 0.855]
SA_Ratio = [0.722, 0.896, 0.928, 0.891, 0.635, 0.734,
            0.751, 0.94, 0.934, 0.908, 0.985]

name = 'SA Ratio'
Metric = []
texts = []
if name == 'ACC':
    Metric = ACC
elif name == 'SRCC':
    Metric = SRCC
elif name == 'PLCC':
    Metric = PLCC
elif name == 'SA Ratio':
    Metric = SA_Ratio
else:
    quit()
# 创建画布，并设置大小和背景颜色
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

for i in range(len(methods)):
    plt.scatter(params[i], Metric[i], label=methods[i], s=300)
    if methods[i] == "Ours":
        plt.annotate(methods[i], (params[i], Metric[i]), ha='center',
                     xytext=(5, -25), textcoords="offset points")

# 添加标题和标签
plt.title(f'Relationship of {name} and Parameters')
ax.set_xlabel('Total Parameters/M')
ax.set_ylabel(f'{name}')
# 显示图形
plt.legend(ncol=2, handleheight=1.5)

plt.show()
