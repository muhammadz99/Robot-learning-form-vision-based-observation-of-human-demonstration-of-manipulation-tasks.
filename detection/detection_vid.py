import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture("vid2.webm")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'),10,size)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    depth = []
    coordinates = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                cx_mid = center_x - 320
                cy_mid = center_y - 320

                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                w_mes = w
                w_real = 3.5

                # # Finding the Focal Length
                #d = 60
                #f = (w_mes*d)/w_real
                #print(f)

                # Finding distance
                f = 528
                d = (w_real * f) / w_mes

                #coordinates
                cx = d
                cy = (cx_mid *w_real)/w_mes
                cz = (cy_mid *w_real)/w_mes

                
                depth.append([class_id,d])
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                coordinates.append([class_id,cx,cy,cz])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

    cv2.imshow('Image', img)
    result.write(img)

    key = cv2.waitKey(1)
    if key==27:
        break
print(coordinates)
cap.release()
cv2.destroyAllWindows()
