import numpy as np
import cv2


class trans():
    def __init__(self, img, pi):
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T

        self.img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
        self.cita = compute_G(self.img_coordinate, pi, height, width)
        self.pi = pi
        self.W, self.A, self.Z = pre_compute_waz(self.pi, height, width, self.img_coordinate)
        self.height = height
        self.width = width

    def deformation(self, img, qi):

        qi = self.pi * 2 - qi
        mapxy = np.swapaxes(np.float32(compute_fv(qi, self.W, self.A, self.Z, self.height, self.width, self.cita, self.img_coordinate)), 0, 1)
        img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP, interpolation=cv2.INTER_LINEAR)

        return img

def pre_compute_waz(pi, height, width, img_coordinate):
    # height*width*控制点个数
    wi = np.reciprocal(np.power(np.linalg.norm(np.subtract(pi, img_coordinate.reshape(height, width, 1, 2)) + 0.000000001, axis=3),2))

    # height*width*2
    pstar = np.divide(np.matmul(wi,pi), np.sum(wi, axis=2).reshape(height,width,1))

    # height*width*控制点个数*2
    phat = np.subtract(pi, pstar.reshape(height, width, 1, 2))

    z1 = np.subtract(img_coordinate, pstar)
    z2 = np.repeat(np.swapaxes(np.array([z1[:,:,1], -z1[:,:,0]]), 1, 2).T.reshape(height,width,1,2,1), [pi.shape[0]], axis=2)

    # height*width*控制点个数*2*1
    z1 = np.repeat(z1.reshape(height,width,1,2,1), [pi.shape[0]], axis=2)

    # height*width*控制点个数*1*2
    s1 = phat.reshape(height,width,pi.shape[0],1,2)
    s2 = np.concatenate((s1[:,:,:,:,1], -s1[:,:,:,:,0]), axis=3).reshape(height,width,pi.shape[0],1,2)

    a = np.matmul(s1, z1)
    b = np.matmul(s1, z2)
    c = np.matmul(s2, z1)
    d = np.matmul(s2, z2)

    # 重构wi形状
    ws = np.repeat(wi.reshape(height,width,pi.shape[0],1),[4],axis=3)

    # height*width*控制点个数*2*2
    A = (ws * np.concatenate((a,b,c,d), axis=3).reshape(height,width,pi.shape[0],4)).reshape(height,width,pi.shape[0],2,2)

    return wi, A, z1

def compute_fv(qi, W, A, Z, height, width, cita, img_coordinate):
    qstar = np.divide(np.matmul(W,qi), np.sum(W, axis=2).reshape(height,width,1))
    qhat = np.subtract(qi, qstar.reshape(height, width, 1, 2)).reshape(height, width, qi.shape[0], 1, 2)
    fv_ = np.sum(np.matmul(qhat, A),axis=2)
    fv = np.linalg.norm(Z[:,:,0,:,:],axis=2) / (np.linalg.norm(fv_,axis=3)+0.0000000001) * fv_[:,:,0,:] + qstar
    fv = (fv - img_coordinate) * cita.reshape(height, width, 1) + img_coordinate
    return fv

def compute_G(img_coordinate, pi, height, width, thre = 0.7):
    max = np.max(pi, 0)
    min = np.min(pi, 0)

    length = np.max(max - min)

    # 计算控制区域中心
    # p_ = (max + min) // 2
    p_ = np.sum(pi,axis=0) // pi.shape[0]

    # 计算控制区域
    minx, miny = min - length
    maxx, maxy = max + length
    minx = minx if minx > 0 else 0
    miny = miny if miny > 0 else 0
    maxx = maxx if maxx < height else height
    maxy = maxy if maxy < width else width

    k1 =(p_ - [0,0])[1] / (p_ - [0,0])[0]
    k2 =(p_ - [height,0])[1] / (p_ - [height,0])[0]
    k4 =(p_ - [0,width])[1] / (p_ - [0,width])[0]
    k3 =(p_ - [height, width])[1] / (p_ - [height, width])[0]
    k = (np.subtract(p_, img_coordinate)[:, :, 1] / (np.subtract(p_, img_coordinate)[:, :, 0] + 0.000000000001)).reshape(height, width, 1)
    k = np.concatenate((img_coordinate, k), axis=2)

    k[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)] = (np.subtract(p_[1], k[:,:,1]) / p_[1]).reshape(height, width, 1)[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)]
    k[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)] = (np.subtract(k[:,:,1], p_[1]) / (width - p_[1])).reshape(height, width, 1)[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)]
    k[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)] = (np.subtract(p_[0], k[:,:,0]) / p_[0]).reshape(height, width, 1)[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)]
    k[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)] = (np.subtract(k[:,:,0], p_[0]) / (height - p_[0])).reshape(height, width, 1)[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)]

    cita = np.exp(-np.power(k[:,:,0] / thre,2))
    cita[minx:maxx,miny:maxy] = 1
    # 如果不需要局部变形，可以把cita的值全置为1
    # cita = 1

    return cita
    
# 里面输入你的图片位置，绝对位置和相对位置都可以
img = cv2.imread('AF1.jpg')
pi = np.array([208, 536, 280, 769, 516, 877, 661, 709, 656, 489]).reshape(-1, 2)
print(pi)
qi = np.array([208, 536, 250, 769, 516, 877, 661, 709, 656, 489]).reshape(-1, 2)

ddd = trans(img, pi)
img = ddd.deformation(img, qi)

cv2.imshow('bianxing', img)
cv2.waitKey(0)


