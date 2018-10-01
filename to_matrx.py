import cv2
import numpy as np

input_image = cv2.imread('bangla2.png', cv2.IMREAD_GRAYSCALE)

def im2bin(i):
    (thresh, im_bw) = cv2.threshold(i, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    h,w = im_bw.shape
    img = im_bw

    img[img==0]=1
    img[img==255]=0
    return img, h, w

img,h,w = im2bin(input_image)

col_mat = np.ones([w,1])
result = np.matmul(img, col_mat)
zero_pos = np.where(result==0)[0]
line_pos =[]

for i in range(0,len(zero_pos)-1):
    if i !=len(zero_pos):
        if  zero_pos[i+1]-zero_pos[i]>1:
            line_pos.append(zero_pos[i])


for pts in line_pos:
    cv2.line(input_image, (0,pts),(w,pts),1)
    print(pts)
cv2.imshow('image',input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(line_pos)
# print(img)
# np.savetxt('output.txt', img)

