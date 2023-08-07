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


# Colorize the region using `colorize()` function.
def colorize(img, coords):
    x1, y1, x2, y2 = coords[:4]
    crop = img[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    red_ch = np.ones((crop_h, crop_w), dtype=np.uint8)*255
    green_ch = np.zeros((crop_h, crop_w), dtype=np.uint8)
    blue_ch = np.zeros((crop_h, crop_w), dtype=np.uint8)
    red_pallet_collection = [blue_ch, green_ch, red_ch]
    red_pallet = cv2.merge(red_pallet_collection)
    colorized_crop = cv2.add(crop, red_pallet)
    img[y1:y2, x1:x2] = colorized_crop

    return img

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
        # Get control region coordinates at 20% margin.
        cr_tlc_x, cr_tlc_y = int(0.20*width), int(0.20*height)
        cr_brc_x, cr_brc_y = int(0.80*width), int(0.80*height)

        cv2.line(canvas, (0, cr_tlc_y), (width,  cr_tlc_y), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, cr_brc_y), (width,  cr_brc_y), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (cr_tlc_x, 0), (cr_tlc_x, height), (0,255,0), 1, cv2.LINE_AA)
        cv2.line(canvas, (cr_brc_x, 0), (cr_brc_x, height), (0,255,0), 1, cv2.LINE_AA)

        # Centroid moved to left region.
        if (xc < cr_tlc_x and cr_tlc_y < yc < cr_brc_y):
            print('Focus moved to left, sending drone left')
            # Glow left region.
            canvas = colorize(canvas, (0, cr_tlc_y, cr_tlc_x, cr_brc_y))
            # send move left command.
        elif (cr_tlc_x < xc < cr_brc_x and yc < cr_tlc_y):
            print('Focus moved to ahead, sending drone forward')
            # Glow forward region
            canvas = colorize(canvas, (cr_tlc_x, 0, cr_brc_x, cr_tlc_y))
            # send move forward command.
        elif (xc > cr_brc_x and cr_tlc_y < yc < cr_brc_y):
            print('Focus moved to right, sending drone right')
            # Glow right region.
            canvas = colorize(canvas, (cr_brc_x, cr_tlc_y, width, cr_brc_y))
            # send move right command.
        elif(cr_tlc_x < xc < cr_brc_x and yc > cr_brc_y):
            print('Focus moved back, sending drone back')
            # Glow bottom region.
            canvas = colorize(canvas, (cr_tlc_x, cr_brc_y, cr_brc_x, height))
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