import cv2
data = [0.396875, 0.275, 0.603125 ,0.615625]

image = cv2.imread('./store/images/frame_0.jpg')
height, width, _ = image.shape
print(height, width)

x1 = int(data[0] * width)
y1 = int(data[1] * height)
x2 = int((data[2]) * width)
y2 = int((data[3]) * height)

image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow('e', image)
cv2.waitKey(0)
