import cv2
# tracker= cv2.TrackerKCF_create()
# tracker = cv2.TrackerCSRT_create()
def create_tracker():
    return cv2.TrackerCSRT_create() 
tracker=create_tracker()
video=cv2.VideoCapture('2932301-uhd_4096_2160_24fps.mp4')
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Make window resizable
cv2.resizeWindow('Video', 800, 600)          # Set desired size (Width, Height)

object_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ok , frame=video.read()
frame = cv2.resize(frame, (800, 600))
bbox=cv2.selectROI('Video',frame,False)
# print(bbox)
ok = tracker.init(frame,bbox)
# print(ok)
while True:
    ok , frame=video.read()
    # print(ok)
    if not ok:
        break
    frame = cv2.resize(frame, (800, 600))
    ok, bbox = tracker.update(frame)
    # print(bbox)
    
    if ok:
        x,y,w,h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y) , (x+w, y+h), (0,255,255),2 ,10)
    else:
        cv2.putText(frame , "lost tracking searching again",(150,100), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(255,0,0) , 2)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = object_cascade.detectMultiScale(gray_frame, 1.9 ,10)
        
        if len(detections) > 0:
            # Assume the largest detected object is the one to track
            x, y, w, h = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)[0]
            
            # Reinitialize the tracker with the detected object
            bbox = (x, y, w, h)
            tracker = create_tracker()
            tracker.init(frame, bbox)
        
        
    
    cv2.imshow('tracking' , frame)
    if cv2.waitKey(1) & 0XFF == 27:
        break        