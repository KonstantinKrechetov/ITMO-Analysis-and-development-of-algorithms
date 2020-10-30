import numpy as np
import matplotlib.pyplot as plt
import math


# Algorithm Graham Scan
def graham_scan(points, start_point, show_progress=False):
    def polar_angel(p_0, p_1=None):
        if p_1 == None: p_1 = start_point
        y_span = p_0[1] - p_1[1]
        x_span = p_0[0] - p_1[0]
        return math.atan2(y_span, x_span)

    def distance(p_0, p_1=None):
        if p_1 == None: p_1 = start_point
        y_span = p_0[1] - p_1[1]
        x_span = p_0[0] - p_1[1]
        return math.sqrt(y_span ** 2 + x_span ** 2)

    def q_sort(arr):
        if len(arr) <= 1: return arr
        smaller, equal, larger = [], [], []
        piv_ang = polar_angel(arr[np.random.randint(0, len(arr) - 1)])
        for point in arr:
            point_ang = polar_angel(point)
            if point_ang < piv_ang:
                smaller.append(point)
            elif point_ang == piv_ang:
                equal.append(point)
            else:
                larger.append(point)
        return q_sort(smaller) + sorted(equal, key=distance) + q_sort(larger)

    def det(p_1, p_2, p_3):
        return (p_2[0] - p_1[0]) * (p_3[1] - p_1[1]) - (p_2[1] - p_1[1]) * (p_3[0] - p_1[0])

    sorted_points = q_sort(points)
    del sorted_points[sorted_points.index(start_point)]
    sorted_points.append(start_point)

    hull = [start_point, sorted_points[0]]
    for point in sorted_points[1:]:

        while det(hull[-2], hull[-1], point) <= 0:
            del hull[-1]
            if len(hull) < 2: break
        hull.append(point)
        if show_progress: drawHull(points, start_point, hull)
    return hull


# Get hull start point
def get_start_point(points):
    y_MIN = min(points, key=lambda t: t[1])[1]
    min_list = [item for item in points if item[1] == y_MIN]
    if len(min_list) > 1:
        return min(min_list, key=lambda x: x[0])
    else:
        return min_list[0]


# Draw Hull
def drawHull(points, hull_name, s_point=None, hull=None):
    fig = plt.figure(figsize=(8, 8))
    plt.style.use('seaborn-darkgrid')
    x, y = zip(*points)
    plt.scatter(x, y, color="grey", s=8)

    if hull != None:
        for i in range(1, len(hull)):
            start_point = hull[i - 1]
            end_point = hull[i]
            plt.scatter(end_point[0], end_point[1], color="black")
            plt.plot((start_point[0], end_point[0]), (start_point[1], end_point[1]), color="blue")

    if start_point != None:
        plt.scatter(s_point[0], s_point[1], color="red", s=36)

    plt.savefig("pictures/" + hull_name + ".png")
    plt.close(fig)


lower_border = 1
upper_border = 100
size = 500

X = np.random.randint(lower_border, upper_border, size)
Y = np.random.randint(lower_border, upper_border, size)
points = list(set([(x_i, y_i) for x_i, y_i in zip(X, Y)]))

start_point = get_start_point(points)

graham_hull = graham_scan(points, start_point)
drawHull(points, "graham", start_point, graham_hull)
