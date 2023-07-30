from ultralytics import YOLO
import cv2 
import numpy as np 


def get_centroid(bboxes):
    tlc_coordinates = []
    brc_coordinates = []
    for box in bboxes:
        coordinates = box.xyxy.numpy()
        x1, y1, x2, y2 = coordinates[0]
        tlc_coordinates.append((x1, y1))
        brc_coordinates.append((x2, y2))
    
    centroid_tlc = np.mean(tlc_coordinates, axis=0)
    centroid_brc = np.mean(brc_coordinates, axis=0)

    xc1, yc1, xc2, yc2 = centroid_tlc[0], centroid_tlc[1], centroid_brc[0], centroid_brc[1]

    return xc1, yc1, xc2, yc2



feed = cv2.VideoCapture('videos/walking-person.mp4')

model = YOLO("yolov8s.pt")
count = 0
while True:
    ret, frame = feed.read()
    count += 1
    if not ret:
        print('Unable to read frames!')
        break
    canvas = frame.copy()
    if count%2 == 0:
        results = model(frame)
        boxes = results[0].boxes
        x_c1, y_c1, x_c2, y_c2 = get_centroid(boxes)
        print('Centroid : ', x_c1, ", ", y_c1, ", ", x_c2, ", ", y_c2)
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