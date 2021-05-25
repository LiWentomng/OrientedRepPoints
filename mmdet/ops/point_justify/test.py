import numpy as np
import torch
from points_justify import pointsJf
import cv2

import numpy as np
import torch
import cv2

points = np.array([[300.,300.],
                   [400.,400.],
                   [100.,100],
                   [300,250],
                   [100,0]])
polygons = np.array([[200.,200.,400.,400.,500.,200.,400.,100.],
                     [400.,400.,500.,500.,600.,300.,500.,200.],
                     [300.,300.,600.,700.,700.,700.,700.,100.]])


# 可视化
canvas_shape = (800,800,3)
for poly_id, polygon in enumerate(polygons):
    for point_id, point in enumerate(points):
        canvas = np.full(canvas_shape,(255,255,255),dtype=np.uint8)
        polygon = np.int0(polygon.reshape(-1,2))
        cv2.drawContours(canvas,[polygon],0,(0,255,0),4)
        # 画圆，圆心为：(160, 160)，半径为：60，颜色为：point_color，实心线
        point = tuple(np.int0(point))
        cv2.circle(canvas, point, 4, (0,0,255), -1)
        save_path = 'images/'+\
            str(poly_id) + '_' + str(point_id) +'.png'
        cv2.imwrite(save_path, canvas)

# 计算所属关系
points = torch.from_numpy(points).cuda().float()
polygons = torch.from_numpy(polygons).cuda().float()
# import ipdb; ipdb.set_trace()
output = torch.full([points.shape[0], polygons.shape[0]], 0.).cuda().float()
pointsJf(points,polygons,output)
print(output)
pass