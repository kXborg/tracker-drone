from ultralytics import YOLO
import cv2 
import numpy as np 
import math

# Find focus point using `get_centroid()` function.
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

# Create video capture object.
feed = cv2.VideoCapture('videos/beach-walk.mp4')

# Load YOLO model.
model = YOLO("yolov8n.pt")

# set counter for frame skipping (may or may not require).
count = 0
while True:
    ret, frame = feed.read()
    # increment counter.
    count += 1
    if not ret:
        print('Unable to read frames!')
        break
    canvas = frame.copy()
    # Get image height width.
    height, width = frame.shape[:2]
    # if count%2 == 0:
    # Filter detected persons.
    results = model.predict(source=frame, classes=0, conf=0.5)
    boxes = results[0].boxes
    # print(boxes)
    
    centroid_coords = get_centroid(boxes)
    # print('Centroid : ', centroid_coords)
    
    if centroid_coords is not None:
        xc, yc = int(centroid_coords[0]), int(centroid_coords[1])
        cv2.circle(canvas, (xc, yc), 4, (0,0,255), -1)

        # Logic for drone control goes here.
        # Get control region coordinates.
        cr_tlc_x, cr_tlc_y = int(0.20*width), int(0.20*height)
        cr_brc_x, cr_brc_y = int(0.80*width), int(0.80*height)

        cv2.line(canvas, (0, cr_tlc_y), (width,  cr_tlc_y), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, cr_brc_y), (width,  cr_brc_y), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (cr_tlc_x, 0), (cr_tlc_x, height), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (cr_brc_x, 0), (cr_brc_x, height), (0,255,0), 1, cv2.LINE_AA)

        # Centroid moved to left region.
        if (xc < cr_tlc_x and cr_tlc_y < yc < cr_brc_y):
            print('Focus moved to left, sending drone left')
            # send move left command.
        elif (cr_tlc_x < xc < cr_brc_x and yc < cr_tlc_y):
            print('Focus moved to ahead, sending drone forward')
            # send move forward command.
        elif (xc > cr_brc_x and cr_tlc_y < yc < cr_brc_y):
            print('Focus moved to right, sending drone right')
            # send move right command.
        elif(cr_tlc_x < xc < cr_brc_x and yc > cr_brc_y):
            print('Focus moved back, sending drone back')
            # send move back command.
        else:
            print('Drone centered')
    
    # for display purpose only, control is up.
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