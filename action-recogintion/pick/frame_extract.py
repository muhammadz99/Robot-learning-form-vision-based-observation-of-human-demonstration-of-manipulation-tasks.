import cv2
vidcap = cv2.VideoCapture('182927.webm')
success, image = vidcap.read()
count = 1
while success:
  cv2.imwrite("p5"+"%d.jpg" % count, image)    
  success, image = vidcap.read()
  print('Saved image ', count)
  count += 1
