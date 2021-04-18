import cv2,os

path = '/home/group00/working/week5/img_task_D'

image = cv2.imread(os.path.join(path,'000041_cat.png'))
mask = cv2.imread(os.path.join(path,'img.png'))

image_output= cv2.imwrite(os.path.join(path,'img_output.png'), cv2.bitwise_and(image,mask))