import cv2
import numpy as np
import matplotlib.pyplot as plt

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml') 
# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def detect_faces(frame):
    face_boxes = face_detector.detectMultiScale(frame)
    if len(face_boxes) == 0:
        raise RuntimeError("No face detected")
    
    return face_boxes

def detect_closest_face(face_boxes, face_coordinates):
    x, y, w, h = face_coordinates
    window = min(face_boxes, key=lambda i: abs(x - i[0]) + abs(y - i[1]))

    return window

def initialize_hue_hist(roi):
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

    # form histogram of hue in the roi
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    print(roi_hist.shape)

    # normalize the histogram array values so they are in the min=0 to max=255 range
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist

def mean_shift_hue(frame, track_window, roi_hist):
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # histogram back projection using roi_hist 
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    # use meanshift to shift the tracking window
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    return track_window

def get_intersection_over_union(window_1, window_2):
    x,y,w,h = window_1
    x2,y2,w2,h2 = window_2

    inter_width = max(min(x + w, x2 + w2) - max(x, x2) + 1, 0)
    inter_height = max(min(y + h, y2 + h2) - max(y, y2) + 1, 0)
    inter_area = inter_width * inter_height

    sum_area = w * h + w2 * h2
    union_area = sum_area - inter_area

    iou = inter_area / union_area
    return iou

def plot_iou_across_frames(ious):
    X = list(range(len(ious)))
    Y = ious

    plt.plot(X, Y, 'm.')
    plt.ylim(0, 1)
    plt.axhline(0.70, color='r', linestyle='dashed', label="t_high")
    plt.axhline(0.60, color='b', linestyle='dashed', label="t_low")
    print(sum(i > 0.70 for i in Y) / len(Y))
    plt.xlabel("Frames over time")
    plt.ylabel("IOU")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # cap = cv2.VideoCapture(0) # camera feed
    cap = cv2.VideoCapture("A4_videos/KylianMbappe.mp4") # video

    # capture one frame
    ret,frame = cap.read()

    # detect a face on the first frame
    face_boxes = detect_faces(frame)

    # initialize the tracing window around the (first) detected face
    (x,y,w,h) = tuple(face_boxes[0]) 
    track_window = (x,y,w,h)
    # rectangle on image (roi)
    roi = frame[y:y+h, x:x+w]

    roi_hist = initialize_hue_hist(roi)

    ious = []

    while True:
        # grab a frame
        ret, frame = cap.read() 
        
        if ret == True: 
            # convert to HSV
            track_window_2 = detect_closest_face(detect_faces(frame), track_window)
            (x2,y2,w2,h2) = track_window_2

            track_window = mean_shift_hue(frame, track_window, roi_hist)
            (x,y,w,h) = track_window

            ious.append(get_intersection_over_union(track_window, track_window_2))

            img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255) ,5) # BGR -> RED
            img = cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (255,0,0), 5) # BGR -> BLUE
            
            cv2.imshow('mean shift tracking demo', img)
            
            if cv2.waitKey(1) & 0xFF == 27: # wait a bit and exit is ESC is pressed
                break
            
        else:
            break
    
    plot_iou_across_frames(ious)
            
    cv2.destroyAllWindows()
    cap.release()