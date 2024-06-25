import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone

model=YOLO('yolov8m.pt') # use 8n, 8s, 8m, 8l, 8x, 8c as per GPU availability

def pointer_location(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('CG Miniproject: People Counter using YOLOv8')
cv2.setMouseCallback('CG Miniproject: People Counter using YOLOv8', pointer_location)

cap=cv2.VideoCapture("vidp1.mp4") # put file name

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
tracker=Tracker(35) # 35 pixel sensitivity of tracker

cy1=250
cy2=300
offset=15 # pixel sensitivity of the counter

uppeople = {}
counteruppeople = []
downpeople = {}
counterdownpeople = []


while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    
    # Extract bounding box data and move to CPU
    a = results[0].boxes.data.cpu().numpy()

    px = pd.DataFrame(a).astype("float")
    people_list=[]
                
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            people_list.append([x1,y1,x2,y2])


    bbox_id = tracker.update(people_list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        # ------------ PEOPLE GOING UP ------------
        if cy1<(cy+offset) and (cy1>cy-offset):
            uppeople[id] = (cx,cy)
        if id in uppeople:
            if cy2<(cy+offset) and (cy2>cy-offset):
                if counteruppeople.count(id) == 0:
                    counteruppeople.append(id)

        # ------------ PEOPLE GOING DOWN ------------
        if cy2<(cy+offset) and (cy2>cy-offset):
            downpeople[id] = (cx,cy)
        if id in downpeople:
            if cy1<(cy+offset) and (cy1>cy-offset):
                if counterdownpeople.count(id) == 0:
                    counterdownpeople.append(id)

        ## See the people getting detected
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
        cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)

    
    cv2.line(frame,(50,cy1),(1020,cy1),(0,255,0),2)
    cv2.line(frame,(30,cy2),(1040,cy2),(0,0,255),2)
    cvzone.putTextRect(frame, f'People coming towards:{len(counteruppeople)}', (50,60), 1,1)
    cvzone.putTextRect(frame, f'People going away:{len(counterdownpeople)}', (50,80), 1,1)
    cv2.imshow("CG Miniproject: People Counter using YOLOv8", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
    
cap.release()
cv2.destroyAllWindows()

