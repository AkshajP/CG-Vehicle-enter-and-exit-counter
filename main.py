import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone

model=YOLO('yolov8m.pt') # use 8n, 8s, 8m, 8l, 8x, 8c as per GPU availability

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture("vidp8.mp4") # put file name


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()


cy1=250
cy2=300
offset=15 # increase to increase the sensitivity of the counter

upcar = {}
counterupcar = []
downcar = {}
counterdowncar = []
upbike = {}
counterupbike = []
downbike = {}
counterdownbike = []
# person not processed yet 
upperson = {}
counterupperson = []
downperson = {}
counterdownperson = []

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
    
    # Create a DataFrame from the numpy array
    px = pd.DataFrame(a).astype("float")
    car_list=[]
    bike_list = []
    person_list = []            
    for index,row in px.iterrows():
        #print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            car_list.append([x1,y1,x2,y2])
        # elif 'motorbike' in c or 'bicycle' in c:
        #     bike_list.append([x1,y1,x2,y2])
        # elif 'person' in c:
        #     person_list.append([x1,y1,x2,y2])


    bbox_id = tracker.update(car_list)
    bbox_id1 = tracker1.update(bike_list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        # ------------ CAR GOING UP ------------
        if cy1<(cy+offset) and (cy1>cy-offset):
            upcar[id] = (cx,cy)
        if id in upcar:
            if cy2<(cy+offset) and (cy2>cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)
                if counterupcar.count(id) == 0:
                    counterupcar.append(id)
        # ------------ CAR GOING DOWN ------------
        if cy2<(cy+offset) and (cy2>cy-offset):
            downcar[id] = (cx,cy)
        if id in downcar:
            if cy1<(cy+offset) and (cy1>cy-offset):
                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)
                if counterdowncar.count(id) == 0:
                    counterdowncar.append(id)

        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
        cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)

    for bbox1 in bbox_id1:
        x3,y3,x4,y4,id = bbox1
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        # ------------ MOTORCYCLE GOING UP ------------
        if cy1<(cy+offset) and (cy1>cy-offset):
            upbike[id] = (cx,cy)
        if id in upcar:
            if cy2<(cy+offset) and (cy2>cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)
                if counterupbike.count(id) == 0:
                    counterupbike.append(id)
        # ------------ MOTORCYCLE GOING DOWN ------------
        if cy2<(cy+offset) and (cy2>cy-offset):
            downbike[id] = (cx,cy)
        if id in downbike:
            if cy1<(cy+offset) and (cy1>cy-offset):
                cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(frame, str(id), (x3,y3), 1,1)
                if counterdownbike.count(id) == 0:
                    counterdownbike.append(id)

    cv2.line(frame,(50,cy1),(1020,cy1),(0,255,0),2)
    cv2.line(frame,(30,cy2),(1040,cy2),(0,0,255),2)
    cvzone.putTextRect(frame, f'upcar:{len(counterupcar)}', (50,60), 1,1)
    cvzone.putTextRect(frame, f'downcar:{len(counterdowncar)}', (50,80), 1,1)
    cvzone.putTextRect(frame, f'upbike:{len(counterupbike)}', (50,100), 1,1)
    cvzone.putTextRect(frame, f'downbike:{len(counterdownbike)}', (50,120), 1,1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

