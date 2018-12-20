import matplotlib.pyplot as plt
import cv2
pathA='./datasets/edges2shirts_stripe/trainB/1.png'
im = cv2.imread(pathA)
im = cv2.cvtColor(im,cv2.COLOR_BayerRG2GRAY)
plt.imshow(im)
plt.show()
