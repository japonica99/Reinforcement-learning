import sys
import cv2
import dlib
import numpy as np

class TriangularWarp:
    def __init__(self,src,dst,img):

        self.img = img
        h, w, c = self.img.shape
        rect = (0, 0, w, h)
        subdiv_src = cv2.Subdiv2D(rect)
        subdiv_dst = cv2.Subdiv2D(rect)
        pair = {}
        for s,d in zip(src,dst):
            subdiv_src.insert((int(s[0]), int(s[1])))
            pair[(int(s[0]), int(s[1]))] = (int(d[0]), int(d[1]))
        bolder = [(0, h - 1),(w / 2, h - 1),(w / 2, 0),(0, h / 2),(w - 1, h / 2),(w - 1, h - 1),(w - 1, 0),(1, 1)]
        for b in bolder:
            subdiv_src.insert(b)
            subdiv_dst.insert(b)
        self.TL_src = subdiv_src.getTriangleList()
        self.TL_dst = subdiv_src.getTriangleList()
        for dstt in self.TL_dst:
            for i in range(3):
                if (dstt[2 * i], dstt[2 * i + 1]) in pair.keys():
                    dstr = pair[(dstt[2 * i], dstt[2 * i + 1])]
                    dstt[2 * i] = dstr[0]
                    dstt[2 * i + 1] = dstr[1]

    def warp(self):
        res = self.img.copy()
        img_copy = self.img.copy()
        for src, dst in zip(self.TL_src, self.TL_dst):

            tri1 = np.float32([[[src[0], src[1]], [src[2], src[3]], [src[4], src[5]]]])
            tri2 = np.float32([[[dst[0], dst[1]], [dst[2], dst[3]], [dst[4], dst[5]]]])
            print(tri1,tri2)
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)
            tri1Cropped = []
            tri2Cropped = []

            for i in range(0, 3):
                tri1Cropped.append(((tri1[0][i][0] - r1[0]), (tri1[0][i][1] - r1[1])))
                tri2Cropped.append(((tri2[0][i][0] - r2[0]), (tri2[0][i][1] - r2[1])))

            M = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
            #M = cv2.getAffineTransform(np.float32(tri1), np.float32(tri2))
            img1Cropped = img_copy[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

            img2Cropped = cv2.warpAffine(img1Cropped, M, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_REFLECT_101)

            # get mask by filling triangle
            mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)
            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask

            # Copy triangular region of the rectangular patch to the output image
            res[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = res[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)

            res[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = res[r2[1]:r2[1] + r2[3],r2[0]:r2[0] + r2[2]] + img2Cropped
        return res

    def draw(self):
        res = self.img.copy()
        for src, dst in zip(self.TL_src, self.TL_dst):
            cv2.line(res, (int(src[0]), int(src[1])), (int(src[2]), int(src[3])), (255, 0, 0), 1)
            cv2.line(res, (int(src[2]), int(src[3])), (int(src[4]), int(src[5])), (255, 0, 0), 1)
            cv2.line(res, (int(src[4]), int(src[5])), (int(src[0]), int(src[1])), (255, 0, 0), 1)
            cv2.line(res, (int(dst[0]), int(dst[1])), (int(dst[2]), int(dst[3])), (0, 0, 255), 1)
            cv2.line(res, (int(dst[2]), int(dst[3])), (int(dst[4]), int(dst[5])), (0, 0, 255), 1)
            cv2.line(res, (int(dst[4]), int(dst[5])), (int(dst[0]), int(dst[1])), (0, 0, 255), 1)
        return res

