import numpy as np
import operator
#创造带有标签的数据集
def create_data():
    x_y=np.array([[10,10],[10,20],[10,30],[60,70],[70,80],[80,90]])
    label_xy=('C','C','C','A','A','A','A')
    return x_y,label_xy
def distance(x_y,label_xy,a,k):
    xy_Size=x_y.shape[0] #读取数据集的数据个数
    distanceMat=np.tile(a,(xy_Size,1))-x_y #实现输入的无标签数据与已有数据集的距离长短（distance=sqrt(x和y的距离差的平方和)）
    sqdistance=distanceMat**2
    sqsum=sqdistance.sum(axis=1)
    xy_distance=sqsum**0.5
    sortedindex=xy_distance.argsort() #实现距离长短的数据集的排序  如[3,2,5]->[1,0,2]
    classcount={}
    for i in range(k):
        vetolabel=label_xy[sortedindex[i]]
        classcount[vetolabel]=classcount.get(vetolabel,0)+1 #统计在最近的k个距离中的数据集的标签
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]#k个数据中最多的标签数
if __name__ == '__main__':
    a,b=create_data()
    c=distance(a,b,[60,60],3)
    print(c)