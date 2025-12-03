import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
import operator
from Rij import weight
from Rij2 import weight2
import xlrd
from matplotlib.patches import Ellipse

G = nx.Graph()  # 创建一个空的无向图
t1 = 3  # 人类驾驶第一反应时间
t2 = 6  # 人类驾驶第二反应时间
lv = 4   # 自车长
wv = 2  # 自车宽
kv = 0.25  # 调节因子
kc = 0.1  # 风险认知调节系数
workbook = xlrd.open_workbook(r'93_8-12-self-data.xls')     # 导入数据
# 获取所有sheet
sheet_name = workbook.sheet_names()[0]

# 根据sheet索引或者名称获取sheet内容
sheet = workbook.sheet_by_index(1)  # sheet索引从0开始
ros = []
for i in range(2):
    rows = sheet.row_values(i)
    ros.append(rows)
data = np.array(ros, dtype=float)

WW = []
E = []

class Graph(object):
    plt.ion()
    plt.axis('equal')
    fig = plt.figure()
    for n in range(56):  
        print(n)
        num = len(data)
        nodes = list(range(num))
        G.add_nodes_from(nodes)  # 从列表中加点 , 将节点添加到网络中
        coordinates = np.array(data[:, 1+4*n:3+4*n])
        coordinates1 = np.array(data[:, 1+4*(n+1):3+4*(n+1)])
        v = data[:, 3+4*n]
        beta = data[:, 4+4*n]
        beta1 = data[:, 4+4*(n+1)]
        if n == 0:
            alpha = data[:, 0]
        else:
            alpha = np.sum([alpha, 1*beta], axis=0)
        alpha1 = np.sum([alpha, 1*beta1], axis=0)
        npos0 = dict(zip(nodes, coordinates))  # 获取节点与坐标之间的映射关系，用字典表示
        npos2 = dict(zip(nodes, coordinates1))  # 下一帧坐标数据
        nlabels = dict(zip(nodes, nodes))  # 标志字典，构建节点与标识点之间的关系
        edges = []  # 存放所有的边
        W = []      # 存放所有权值

        lv1 = lv+2*kv*v[0]*math.cos(beta[0]/180*math.pi)
        wv1 = wv+2*kv*v[0]*math.sin(beta[0]/180*math.pi)
        rp = []
        for i in range(2):
            x0, y0 = npos0[i]
            x = (x0-npos0[0][0])*math.cos((360-alpha[0])/180*math.pi)-(y0-npos0[0][1])*math.sin((360-alpha[0])/180*math.pi)
            y = (x0-npos0[0][0])*math.sin((360-alpha[0])/180*math.pi)+(y0-npos0[0][1])*math.cos((360-alpha[0])/180*math.pi)
            rp.append((x, y))
            npos1 = dict(zip(nodes, rp))
        rp2 = []
        for i in range(2):
            x0, y0 = npos2[i]
            x = (x0-npos2[0][0])*math.cos((360-alpha1[0])/180*math.pi)-(y0-npos2[0][1])*math.sin((360-alpha1[0])/180*math.pi)
            y = (x0-npos2[0][0])*math.sin((360-alpha1[0])/180*math.pi)+(y0-npos2[0][1])*math.cos((360-alpha1[0])/180*math.pi)
            rp2.append((x, y))
            npos3 = dict(zip(nodes, rp2))

        a0 = t1*v[0]  # 存在问题，第一认知域椭圆的长轴
        b0 = a0/(lv1/wv1)  # 第一认知域椭圆的短轴
        a1 = t2*v[0]
        b1 = a1/(lv1/wv1)
        w1 = []  # 第一认知域内自车与环境节点权值

        ax = fig.add_subplot(111)
        ell1 = Ellipse(xy=(npos0[0]), width=a0*2, height=b0*2, angle=alpha[0], facecolor='r')
        ell2 = Ellipse(xy=(npos0[0]), width=a1 * 2, height=b1 * 2, angle=alpha[0], facecolor='yellow')
        ax.add_patch(ell2)
        ax.add_patch(ell1)

        # bod = ((a1, 0), (a1 * -1, 0), (0, b1), (0, b1 * -1))
        # for i in bod:
        #     x0, y0 = i
        #     x = x0*math.cos((alpha[0])/180*math.pi)-y0*math.sin((alpha[0])/180*math.pi)+npos0[0][0]
        #     y = x0*math.sin((alpha[0])/180*math.pi)+y0*math.cos((alpha[0])/180*math.pi)+npos0[0][1]
        #     plt.scatter(x, y, marker='x', color='red', s=40)

        # image = np.zeros((500, 500, 3), np.uint8)  # 创建一个黑色面板
        # cv2.ellipse(image, (int(250+npos0[0][0]), int(250+npos0[0][1])), (int(a0), int(b0)), alpha[0],  0,360,(255, 0, 0))  # 画半个椭圆
        # cv2.ellipse(image, (int(250+npos0[0][0]), int(250+npos0[0][1])), (int(a1), int(b1)), alpha[0], 0,360,(255, 255, 0))  # 画45度角的整个椭圆
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 色彩空间转换
        # cv2.imshow("绘制椭圆", image)

        i = []   #存储满足条件的节点编号
        w = []  # 存储满足条件的权重
        for idx in range(1, num):
            x0, y0 = npos1[0]
            x, y = npos1[idx]
            if pow(x, 2)/pow(a0, 2) + pow(y, 2)/pow(b0, 2) <= 1: #如果节点在椭圆内
                i.append(idx) # 将编号加入i内
                R = weight(npos3[idx], npos1[idx], beta[0], v[0], v[idx])   # 计算边权值
                w1.append([idx, R])   # 将索引和边权值加入w1
                w.append(R)           #  将边权值加入w
        wn = dict(zip(i, w))     # 将索引和边权值一一对应，打包加入wn
        w2 = sorted(w1, key=operator.itemgetter(1), reverse=True)
        # 第一认知域内自车与环境节点权值排序
        # 调用 sorted() 函数对列表 w1 进行排序，按照节点权重 R 从大到小排序，并将排序结果保存在列表 w2 中
        for x in w2:
            edges.append((0,x[0]))
            W.append(x[1])
        # print(W)  # 第一步结束

        for x in w2:
            x0, y0 = npos0[x[0]]
            w3 = []                    # 第一认知域内自车与环境节点权值
            i = x[0]
            print("第一认知域内节点序列", i)
            for y in w2:
                j = y[0]
                x, y = npos0[j]
                x2, y2 = npos2[j]     # 环境节点下一帧原坐标
                if i == j:
                    continue
                else:
                    x1 = (x - x0) * math.cos((360 - alpha[i]) / 180 * math.pi) - (y - y0) * math.sin(
                        (360 - alpha[i]) / 180 * math.pi)
                    y1 = (x - x0) * math.sin((360 - alpha[i]) / 180 * math.pi) + (y - y0) * math.cos(
                        (360 - alpha[i]) / 180 * math.pi)
                    x3 = (x2 - npos2[i][0]) * math.cos((360 - alpha1[i]) / 180 * math.pi) - (
                                y2 - npos2[i][1]) * math.sin((360 - alpha1[i]) / 180 * math.pi)
                    y3 = (x2 - npos2[i][0]) * math.sin((360 - alpha1[i]) / 180 * math.pi) + (
                                y2 - npos2[i][1]) * math.cos((360 - alpha1[i]) / 180 * math.pi)
                    if i == 8 or i == 9:
                        R = weight2((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                    else:
                        R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                    w3.append([j, R])
            w4 = sorted(w3, key=operator.itemgetter(1), reverse=True)
            if not w4:
                continue
            else:
                edges.append((i,w4[0][0]))
                W.append(w4[0][1])
        # print(W)    #  第二步结束，第一认知域结束

        if not w2:   # 第一认知域没车
            print('第一认知域没车')
            for idx in range(1, num):
                x0, y0 = npos1[0]
                x, y = npos1[idx]
                # print(idx, pow(x, 2) / pow(a1, 2) + pow(y, 2) / pow(b1, 2), x*x/a1/a1+y*y/b1/b1)
                # c = (a1**2-b1**2)**0.5
                # if ((x-c)**2+y**2)**0.5 + ((x+c)**2+y**2)**0.5 <= 2*a1:
                # if ((x*math.cos(alpha[0]/180*math.pi))-(y*math.sin(alpha[0]/180*math.pi)))**2/(a1**2)+((x*math.sin(alpha[0]/180*math.pi))+(y*math.cos(alpha[0]/180*math.pi)))**2/(b1**2)<1:
                if x*x/a1/a1+y*y/b1/b1 <= 1:
                    R = weight(npos3[idx], npos1[idx], beta[0], v[0], v[idx])
                    w1.append([idx, R])
            w2 = sorted(w1, key=operator.itemgetter(1), reverse=True)  # 第一认知域内自车与环境节点权值排序
            for x in w2:
                edges.append((0,x[0]))
                W.append(x[1])
            # print(W)

            for x in w2:
                x0, y0 = npos0[x[0]]
                w3 = []                    # 第二认知域内自车与环境节点权值
                i = x[0]
                print("第二认知域内节点序列", i)
                for y in w2:
                    j = y[0]
                    x, y = npos0[j]
                    x2, y2 = npos2[j]  # 环境节点下一帧原坐标
                    if i == j:
                        continue
                    else:

                        x1 = (x - x0) * math.cos((360 - alpha[i]) / 180 * math.pi) - (y - y0) * math.sin(
                            (360 - alpha[i]) / 180 * math.pi)
                        y1 = (x - x0) * math.sin((360 - alpha[i]) / 180 * math.pi) + (y - y0) * math.cos(
                            (360 - alpha[i]) / 180 * math.pi)
                        x3 = (x2 - npos2[i][0]) * math.cos((360 - alpha1[i]) / 180 * math.pi) - (
                                y2 - npos2[i][1]) * math.sin((360 - alpha1[i]) / 180 * math.pi)
                        y3 = (x2 - npos2[i][0]) * math.sin((360 - alpha1[i]) / 180 * math.pi) + (
                                y2 - npos2[i][1]) * math.cos((360 - alpha1[i]) / 180 * math.pi)
                        if i == 8 or i == 9:
                            R = weight2((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        else:
                            R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])    # 计算权值
                        w3.append([j, R])
                w4 = sorted(w3, key=operator.itemgetter(1), reverse=True)
                if w4:
                    edges.append((i,w4[0][0]))
                    W.append(w4[0][1])
            # print(W)
        else:
            print('第一认知域有车')
            w6 = []
            for i in range(1, num):  # i第二认知域节点
                x0, y0 = npos1[i]  # i的新坐标，用来判断是否为第二认知域内的点
                x, y = npos0[i]  # i的旧坐标
                w5 = []  # 终点与中间点的权值和
                if (x0**2/a1**2 + y0**2/b1**2) <= 1 and (x0**2/a0**2 + y0**2/b0**2) > 1 :
                    print('第二认知域内的点', i)
                    for j in range(1, num):  # j中间节点
                        if i == j:
                            continue
                        else:
                            xp, yp = npos0[j]  # j的旧坐标
                            xq, yq = npos1[j]  # j的新坐标，用来判断是否为第一认知域内的节点
                            x2, y2 = npos2[j]  # 环境节点下一帧原坐标
                            if pow(xq, 2) / pow(a0, 2) + pow(yq, 2) / pow(b0, 2) <= 1 and y0*yq > 0:
                                x1 = (xp - x) * math.cos((360 - alpha[i]) / 180 * math.pi) - (yp - y) * math.sin(
                                    (360 - alpha[i]) / 180 * math.pi)
                                y1 = (xp - x) * math.sin((360 - alpha[i]) / 180 * math.pi) + (yp - y) * math.cos(
                                    (360 - alpha[i]) / 180 * math.pi)
                                x3 = (x2 - npos2[i][0]) * math.cos((360 - alpha1[i]) / 180 * math.pi) - (
                                        y2 - npos2[i][1]) * math.sin((360 - alpha1[i]) / 180 * math.pi)
                                y3 = (x2 - npos2[i][0]) * math.sin((360 - alpha1[i]) / 180 * math.pi) + (
                                        y2 - npos2[i][1]) * math.cos((360 - alpha1[i]) / 180 * math.pi)
                                if i == 8 or i == 9:
                                    R = weight2((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                                else:
                                    R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                                R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                                # m = list(wn.keys())
                                # if j in m:
                                z2 = R + wn[j]
                                w5.append([j, z2])  # 权值和=自车与中间节点权值+中间节点与终点权值
                    # print(sorted(w5,key=lambda x: (-x[1], -x[1]))[0][0])
                    if not w5:
                        R = weight(npos3[i], npos1[i], beta[0], v[0], v[i])
                        w5.append([0, R])
                    if w5:
                        w6.append([i, sorted(w5, key=operator.itemgetter(1), reverse=True)[0][0]])

            for x in w6:
                x0, y0 = npos0[x[0]]
                xp, yp = npos1[x[0]]
                w7 = []                    # 二认知域内节点与第一认知域内节点权值
                w9 = []                    # 第二认知域内节点与第二认知域内节点权值
                i = x[0]                  # 第二认知域内节点的标签
                print("第二认知域内节点序列", i)
                for y in w2:
                    j = y[0]             # 第一认知域内节点的标签
                    x, y = npos0[j]
                    xq, yq = npos1[j]
                    x2, y2 = npos2[j]  # 环境节点下一帧原坐标
                    if yp*yq > 0:
                        x1 = (x - x0) * math.cos((360 - alpha[i]) / 180 * math.pi) - (y - y0) * math.sin(
                            (360 - alpha[i]) / 180 * math.pi)
                        y1 = (x - x0) * math.sin((360 - alpha[i]) / 180 * math.pi) + (y - y0) * math.cos(
                            (360 - alpha[i]) / 180 * math.pi)
                        x3 = (x2 - npos2[i][0]) * math.cos((360 - alpha1[i]) / 180 * math.pi) - (
                                y2 - npos2[i][1]) * math.sin((360 - alpha1[i]) / 180 * math.pi)
                        y3 = (x2 - npos2[i][0]) * math.sin((360 - alpha1[i]) / 180 * math.pi) + (
                                y2 - npos2[i][1]) * math.cos((360 - alpha1[i]) / 180 * math.pi)
                        if i == 8 or i == 9:
                            R = weight2((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        else:
                            R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        w7.append([j, R])
                if not w7:
                    R = weight(npos3[i], npos1[i], beta[0], v[0], v[i])
                    w7.append([0, R])
                if w7:
                    w8 = sorted(w7, key=operator.itemgetter(1), reverse=True)
                if w8:
                    edges.append((i,w8[0][0]))
                    W.append(w8[0][1])

                for y in w6:
                    j = y[0]           # 第二认知域内节点的标签
                    x, y = npos0[j]
                    x2, y2 = npos2[j]  # 环境节点下一帧原坐标
                    if j == i:
                        continue
                    else:
                        x1 = (x - x0) * math.cos((360 - alpha[i]) / 180 * math.pi) - (y - y0) * math.sin(
                            (360 - alpha[i]) / 180 * math.pi)
                        y1 = (x - x0) * math.sin((360 - alpha[i]) / 180 * math.pi) + (y - y0) * math.cos(
                            (360 - alpha[i]) / 180 * math.pi)
                        x3 = (x2 - npos2[i][0]) * math.cos((360 - alpha1[i]) / 180 * math.pi) - (
                                y2 - npos2[i][1]) * math.sin((360 - alpha1[i]) / 180 * math.pi)
                        y3 = (x2 - npos2[i][0]) * math.sin((360 - alpha1[i]) / 180 * math.pi) + (
                                y2 - npos2[i][1]) * math.cos((360 - alpha1[i]) / 180 * math.pi)
                        if i == 8 or i == 9:
                            R = weight2((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        else:
                            R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        R = weight((x3, y3), (x1, y1), beta[i], v[i], v[j])  # 计算权值
                        w9.append([j, R])
                w10 = sorted(w9, key=operator.itemgetter(1), reverse=True)
                if w10:
                    edges.append((i,w10[0][0]))
                    W.append(w10[0][1])

        # edges.to_excle('91-w.xls', index=False)
        # W.to_excle('91-w.xls', index=False)


        # f = open('91-w.xls', 'a')

        # print(edges, file=f)
        # print((W, n), file=f)
        WW.append((n, edges, W))

        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight=W[i])

        # G.add_edges_from(edges)        #  将所有边加入网络
        # G.add_weighted_edges_from((edges.edges()),W)
        nx.draw_networkx_nodes(G, npos0, node_size=200, node_color="#6CB6FF")
        nx.draw_networkx_edges(G, npos0, edges, width=2.0,)  # 绘制边
        nx.draw_networkx_labels(G, npos0, nlabels)  # 标签
        # nx.draw_networkx_edge_labels(G, npos0, edge_labels=None, label_pos=0.5,  font_color='k')



    #坐标轴美化

        x_max, y_max = (20, 20)  # 获取每一列最大值
        x_min, y_min = (-20, -30)  # 获取每一列最小值
        x_num = (x_max - x_min) / 10
        y_num = (y_max - y_min) / 10
        # print(x_max, y_max, x_min, y_min)
        plt.xlim(x_min - x_num, x_max + x_num)
        plt.ylim(y_min - y_num, y_max + y_num)

        plt.pause(0.5)
        plt.clf()
        # plt.close()

        n += 1
    print(WW)
    plt.ioff()


