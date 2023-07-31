from ultralytics import YOLO
import cv2 
import numpy as np 
import math

def get_centroid(bboxes):
    # print('Bboxes: ', bboxes)
    if len(bboxes.xyxy.numpy()) != 0:
        tlc_coordinates = []
        brc_coordinates = []
        for box in bboxes:
            coordinates = box.xyxy.numpy()
            x1, y1, x2, y2 = coordinates[0]
            tlc_coordinates.append((x1, y1))
            brc_coordinates.append((x2, y2))
        
        centroid_tlc = np.mean(tlc_coordinates, axis=0)
        centroid_brc = np.mean(brc_coordinates, axis=0)
        centroid = np.mean((centroid_tlc, centroid_brc), axis=0)

        return centroid


feed = cv2.VideoCapture('videos/walking-person.mp4')

model = YOLO("yolov8n.pt")

count = 0
while True:
    ret, frame = feed.read()
    count += 1
    if not ret:
        print('Unable to read frames!')
        break
    canvas = frame.copy()
    # if count%2 == 0:
    results = model.predict(source=frame, conf=0.5)
    boxes = results[0].boxes
    
    centroid_coords = get_centroid(boxes)
    print('Centroid : ', centroid_coords)
    
    if centroid_coords is not None:
        xc, yc = int(centroid_coords[0]), int(centroid_coords[1])
        cv2.circle(canvas, (xc, yc), 4, (0,0,255), -1)

    height, width = frame.shape[:2]
    
    for box in boxes:
        coordinates = box.xyxy.numpy()
        x1, y1, x2, y2 = coordinates[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
        # print(x1, ',', x2, ',', y1, ',', y2)

    cv2.imshow("Results", cv2.resize(canvas, None, fx=0.5, fy=0.5))

    key = cv2.waitKey(1)

    if key == ord('q'):
        break 
feed.release()
cv2.destroyAllWindows()