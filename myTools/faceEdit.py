import cv2
import dlib
import numpy as np
import math

def bilinear_interpolation(img,vector_u,c):
    ux,uy=vector_u
    x1,x2 = int(ux),int(ux+1)
    y1,y2 = int(uy),int(uy+1)

    f_x_y1 = (x2-ux)/(x2-x1)*img[y1][x1][c]+(ux-x1)/(x2-x1)*img[y1][x2][c]
    f_x_y2 = (x2 - ux) / (x2 - x1) * img[y2][x1][c] + (ux - x1) / (x2 - x1) * img[y2][x2][c]

    f_x_y = (y2-uy)/(y2-y1)*f_x_y1+(uy-y1)/(y2-y1)*f_x_y2
    return int(f_x_y)


def Local_scaling_warps(img,cx,cy,r_max,a):
    img1 = np.copy(img)
    for y in range(cy-r_max,cy+r_max+1):
        d = int(math.sqrt(r_max**2-(y-cy)**2))
        x0 = cx-d
        x1 = cx+d
        for x in range(x0,x1+1):
            r = math.sqrt((x-cx)**2 + (y-cy)**2)
            for c in range(3):
                if r<=r_max:
                    vector_c = np.array([cx, cy])
                    vector_r =np.array([x,y])-vector_c
                    f_s = (1-((r/r_max-1)**2)*a)
                    vector_u = vector_c+f_s*vector_r#原坐标
                    img1[y][x][c] = bilinear_interpolation(img,vector_u,c)
    return img1


def extract_feature_points(image_path):
    # Instantiate the face detector and landmark predictor.
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Load the image.
    image = dlib.load_rgb_image(image_path)

    # Detect faces in the image.
    face_rects = detector(image, 1)

    # Extract feature point locations for each detected face.
    feature_points_list = []
    for face_rect in face_rects:
        shape = predictor(image, face_rect)
        feature_points = [np.array([shape.part(i).x, shape.part(i).y]) for i in range(68)]
        feature_points_list.append(feature_points)

    return feature_points_list

def big_eye(img,r_max,a,left_eye_pos=None,right_eye_pos=None):
    img = Local_scaling_warps(img,left_eye_pos[0],left_eye_pos[1],r_max=r_max,a=a)
    img = Local_scaling_warps(img,right_eye_pos[0],right_eye_pos[1],r_max=r_max,a=a)
    return img

feature_points_list = extract_feature_points("AF1.jpg")
left_eye_pos = (feature_points_list[0][36] + feature_points_list[0][39]) / 2
right_eye_pos = (feature_points_list[0][42] + feature_points_list[0][45]) / 2
r_max = np.linalg.norm(feature_points_list[0][36] - feature_points_list[0][39]).astype(np.int16)
print(r_max)
img = big_eye(cv2.imread("AF1.jpg"),r_max=r_max,a=0.6,left_eye_pos=left_eye_pos.astype(np.int16),right_eye_pos=right_eye_pos.astype(np.int16))
cv2.circle(img, (int(left_eye_pos[0]), int(left_eye_pos[1])), 5, (0, 0, 255), -1)
cv2.circle(img, (int(right_eye_pos[0]), int(right_eye_pos[1])), 5, (0, 0, 255), -1)
cv2.imshow("img",img)
cv2.waitKey(0)

