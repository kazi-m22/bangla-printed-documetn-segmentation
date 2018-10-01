#take one line then segment word :)
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image = cv2.imread('bangla.png', cv2.IMREAD_GRAYSCALE)

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

line1 = img[line_pos[0]:line_pos[1],0:w]
line1_h = line_pos[1]-line_pos[0]

row_mat = np.ones([1,line1_h])
result2 = np.matmul(row_mat,line1)
word_zero_pos = np.where(result2 == 0)[1]
print(word_zero_pos)

def dif(n):
    differences = []
    for i in range(0,len(n)-1):
        differences.append(n[i+1]-n[i])

    return differences

print(dif(word_zero_pos))






