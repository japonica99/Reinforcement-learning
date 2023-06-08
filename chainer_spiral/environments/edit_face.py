try:
    from .face_tools import ImageTransformer,Mls_deformer,TriangularWarp
except:
    from face_tools import ImageTransformer,Mls_deformer,TriangularWarp

import cv2
import numpy as np
import dlib
import math
from copy import deepcopy
#from pychubby.base import DisplacementField


def nothing(x):
    pass

class trans():
    def __init__(self, img, pi):
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T

        self.img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
        #self.cita = compute_G(self.img_coordinate, pi, height, width)
        self.pi = pi
        self.W, self.A, self.Z = pre_compute_waz(self.pi, height, width, self.img_coordinate)
        self.height = height
        self.width = width

    def deformation(self, img, qi):

        qi = self.pi * 2 - qi
        mapxy = np.swapaxes(np.float32(compute_fv(qi, self.W, self.A, self.Z, self.height, self.width, self.img_coordinate)), 0, 1)
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

def compute_fv(qi, W, A, Z, height, width, img_coordinate):
    qstar = np.divide(np.matmul(W,qi), np.sum(W, axis=2).reshape(height,width,1))
    qhat = np.subtract(qi, qstar.reshape(height, width, 1, 2)).reshape(height, width, qi.shape[0], 1, 2)
    fv_ = np.sum(np.matmul(qhat, A),axis=2)
    fv = np.linalg.norm(Z[:,:,0,:,:],axis=2) / (np.linalg.norm(fv_,axis=3)+0.0000000001) * fv_[:,:,0,:] + qstar
    #fv = (fv - img_coordinate) * cita.reshape(height, width, 1) + img_coordinate
    return fv


def relu(x):
    x_ = x.copy()
    x_[x_<0] = 0
    return x_

class AdjustEyeSize():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["eyesize"]

    def __call__(self, list_editted, parameters,feature_list):
        editted = list_editted[0]
        left_eye_pos = np.add(feature_list[0][36], feature_list[0][39]) / 2
        right_eye_pos = np.add(feature_list[0][42], feature_list[0][45]) / 2
        r_max = int(np.linalg.norm(np.subtract(feature_list[0][36], feature_list[0][39])))
        editted_ = self.Local_scaling_warps(editted, int(left_eye_pos[0]), int(left_eye_pos[1]), r_max=r_max, a=0.5 * parameters[0])
        editted_ = self.Local_scaling_warps(editted_, int(right_eye_pos[0]), int(right_eye_pos[1]), r_max=r_max, a=0.5 * parameters[0])
        return [editted_]

    def bilinear_interpolation(self,img, vector_u, c):
        ux, uy = vector_u
        x1, x2 = int(ux), int(ux + 1)
        y1, y2 = int(uy), int(uy + 1)

        f_x_y1 = (x2 - ux) / (x2 - x1) * img[y1][x1][c] + (ux - x1) / (x2 - x1) * img[y1][x2][c]
        f_x_y2 = (x2 - ux) / (x2 - x1) * img[y2][x1][c] + (ux - x1) / (x2 - x1) * img[y2][x2][c]

        f_x_y = (y2 - uy) / (y2 - y1) * f_x_y1 + (uy - y1) / (y2 - y1) * f_x_y2
        return int(f_x_y)

    def Local_scaling_warps(self,img, cx, cy, r_max, a):
        img1 = np.copy(img)
        for y in range(cy - r_max, cy + r_max + 1):
            d = int(math.sqrt(r_max ** 2 - (y - cy) ** 2))
            x0 = cx - d
            x1 = cx + d
            for x in range(x0, x1 + 1):
                r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                for c in range(3):
                    if r <= r_max:
                        vector_c = np.array([cx, cy])
                        vector_r = np.array([x, y]) - vector_c
                        f_s = (1 - ((r / r_max - 1) ** 2) * a)
                        vector_u = vector_c + f_s * vector_r  # 原坐标
                        img1[y][x][c] = self.bilinear_interpolation(img, vector_u, c)
        return img1

class AdjustNoseHeight():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["noseheight"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        for i in range(31,36):
            feature_list_[0][i][1] = int(feature_list[0][i][1] + 0.2 * parameters[0] * (feature_list[0][i][1] - feature_list[0][30][1]))
        return feature_list_

class AdjustNoseWidth():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["nosewidth"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        for i in range(31,36):
            feature_list_[0][i][0] = int(feature_list[0][i][0] + 0.2 * parameters[0] * (feature_list[0][i][0] - feature_list[0][33][0]))
        return feature_list_

class AdjustUpperLipHeight():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["upperlipheight"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        height = feature_list[0][51][1] - feature_list[0][57][1]
        for i in range(48,55):
           feature_list_[0][i][1] = int(feature_list[0][i][1] + 0.1 * parameters[0] * height)
        return feature_list_

class AdjustDownLipHeight():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["downlipheight"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        height = feature_list[0][51][1] - feature_list[0][57][1]
        for i in range(55,60):
           feature_list_[0][i][1] = int(feature_list[0][i][1] - 0.2 * parameters[0] * height)
        return feature_list_

class AdjustMouthHeight():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["mouthheight"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        height = feature_list[0][51][1] - feature_list[0][57][1]
        for i in range(48,68):
           feature_list_[0][i][1] = int(feature_list[0][i][1] + 0.2 * parameters[0] * height)
        return feature_list_

class AdjustMouthWidth():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["mouthwidth"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        for i in range(48,68):
           feature_list_[0][i][0] = int(feature_list[0][i][0] + 0.1 * parameters[0] * (feature_list[0][i][0] - feature_list[0][66][0]))

        return feature_list_

class AdjustChinWidth():
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["chinwidth"]

    def __call__(self, parameters,feature_list):
        feature_list_ = deepcopy(feature_list)
        for i in range(0,17):
            feature_list_[0][i][0] = int(feature_list[0][i][0] - 0.15 * parameters[0] * (feature_list[0][i][0] - feature_list[0][34][0]))

        return feature_list_

class FaceEditor():
    def __init__(self):
        self.edit_funcs = [AdjustEyeSize()]
        self.edit_landmarks_funcs = [AdjustNoseHeight(),AdjustNoseWidth(), AdjustUpperLipHeight(),AdjustDownLipHeight(),AdjustMouthHeight(),AdjustMouthWidth(),AdjustChinWidth()]
        #self.edit_landmarks_funcs = [AdjustChinWidth()]
        self.num_parameters = 0
        for edit_func in self.edit_funcs:
            self.num_parameters += edit_func.num_parameters
        for edit_landmarks_func in self.edit_landmarks_funcs:
            self.num_parameters += edit_landmarks_func.num_parameters

    def __call__(self, photo, parameters,feature_points_list,**interpolation_kwargs):
        output_list = [photo]
        num_parameters = 0

        for edit_func in self.edit_funcs:
            output_list = edit_func(output_list,
                parameters[num_parameters: num_parameters + edit_func.num_parameters],
                                    feature_points_list)
            num_parameters = num_parameters + edit_func.num_parameters

        original_landmarks = deepcopy(feature_points_list[0])
        #mls = trans(output_list[0], np.array(feature_points_list[0]))
        self.feature_points_list = feature_points_list
        for edit_landmarks_func in self.edit_landmarks_funcs:
            self.feature_points_list = edit_landmarks_func(parameters[num_parameters: num_parameters + edit_landmarks_func.num_parameters],
                                    self.feature_points_list)

            num_parameters = num_parameters + edit_landmarks_func.num_parameters


        #用原图的landmarks和变换后的landmarks做mls变换
        #img = mls.deformation(output_list[0], np.array(self.feature_points_list[0]))

        # trans = ImageTransformer(output_list[0], np.array(original_landmarks)[:,[1,0]],np.array(self.feature_points_list[0])[:,[1,0]],color_dim=2, interp_order=2,extrap_mode='nearest')
        # img = trans.deform_viewport()

        #注意这边这样clip只能针对正方形输入
        diangle = TriangularWarp(np.clip(np.array(original_landmarks),0,output_list[0].shape[0] - 1),
                                 np.clip(np.array(self.feature_points_list[0]),0,output_list[0].shape[0] - 1),
                                 output_list[0])
        img = diangle.warp()
        #img = diangle.draw()

        # deformer = Mls_deformer(output_list[0])
        # img = deformer.trans(np.array(original_landmarks)[:,[1,0]],np.array(self.feature_points_list[0])[:,[1,0]],mode='affine')

        # if not interpolation_kwargs:
        #     interpolation_kwargs = {"function": "linear"}
        #
        # df = DisplacementField.generate(output_list[0].shape[:2],
        #                                 np.array(original_landmarks)[:, [1, 0]],
        #                                 np.array(self.feature_points_list[0])[:, [1, 0]],
        #                                 anchor_corners=True,
        #                                 **interpolation_kwargs
        #                                 )
        # output_list[0] = df.warp(output_list[0])


        # for i,p in enumerate(self.feature_points_list[0]):
        #     print(i)
        #     p1 = original_landmarks[i]
        #     cv2.line(img, (p[0], p[1]), (p1[0], p1[1]), (0, 255, 0), 1)
        #     #cv2.circle(img, (p[0], p[1]), 2, (0, 255, 0), -1)

        # return img
        return img

    def extract_feature(self,photo,newsize):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Detect faces in the image.
        face_rects = detector(photo, 1)

        # Extract feature point locations for each detected face.
        feature_points_list = []
        for face_rect in face_rects:
            shape = predictor(photo, face_rect)
            feature_points = [[int(shape.part(i).x / photo.shape[0] * (newsize[0] - 1)), int(shape.part(i).y / photo.shape[1] * (newsize[1] - 1))] for i in range(68)]
            feature_points_list.append(feature_points)

        return feature_points_list




def edit_demo(photo, parameters_=None):
    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.namedWindow('parameter', cv2.WINDOW_NORMAL)

    parameter_dammy = np.ones((1,400,3)) * (233 / 255)

    #input = photo / 255.0
    face_editor = FaceEditor()

    if parameters_ is None:
        parameters = np.zeros(face_editor.num_parameters)
    else:
        parameters = parameters_.copy()

    j = 0
    for edit_func in face_editor.edit_funcs:
        for i in range(edit_func.num_parameters):
            cv2.createTrackbar(edit_func.slider_names[i],edit_func.window_names[i],int((parameters[j]+1)*100),200,nothing)
            j += 1
    for edit_landmarks_func in face_editor.edit_landmarks_funcs:
        for i in range(edit_landmarks_func.num_parameters):
            cv2.createTrackbar(edit_landmarks_func.slider_names[i],edit_landmarks_func.window_names[i],int((parameters[j]+1)*100),200,nothing)
            j += 1

    print("[[Press esc to quit.]]")
    while(1):
        parameters = []
        for edit_func in face_editor.edit_funcs:
            for i in range(edit_func.num_parameters):
                parameters.append(cv2.getTrackbarPos(edit_func.slider_names[i],edit_func.window_names[i])/100.0-1)
        for edit_landmarks_func in face_editor.edit_landmarks_funcs:
            for i in range(edit_landmarks_func.num_parameters):
                parameters.append(cv2.getTrackbarPos(edit_landmarks_func.slider_names[i],edit_landmarks_func.window_names[i])/100.0-1)

        feature_points_list = face_editor.extract_feature(photo.copy(),photo.shape[:2])
        output = face_editor(photo.copy(), parameters,feature_points_list)
        cv2.imshow('face', np.hstack([photo, output]))
        cv2.imshow('parameter',parameter_dammy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


if __name__=="__main__":
    parameters = np.array([0.125,0.5,0.5,0.3,0.3,0.3,0.5,0.5])
    #parameters = np.array([0.5,0.5,0.5])
    image = cv2.imread("../../scutfbp5500_dataset/origin/AF1.jpg")
    #image = cv2.imread("../../images/1039.png")
    edit_demo(image, parameters)
