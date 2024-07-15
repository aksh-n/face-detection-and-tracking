import cv2
import numpy as np
import matplotlib.pyplot as plt

from mean_shift_hue import detect_faces, detect_closest_face, plot_iou_across_frames, get_intersection_over_union

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml') 
# termination criteria for mean shift: 10 iteration or shift less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def get_combined_grad(gray, kernel_size):
    gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel_size)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel_size)

    mag, ang = cv2.cartToPolar(gX, gY, angleInDegrees=True)
    combined_grad = np.dstack((mag, ang))
    
    return combined_grad

def initialize_grad_hist(roi, kernel_size=3):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5,5), 0)
    combined_grad = get_combined_grad(gray_roi, kernel_size)
    max_grad = np.max(combined_grad)
    mask = cv2.inRange(combined_grad, np.array((0.1 * max_grad, 0.)), np.array((1. * max_grad, 360.0)))

    roi_hist = cv2.calcHist([combined_grad],[1],mask,[24],[0,max_grad])
    # normalize the histogram array values so they are in the min=0 to max=255 range
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist

def mean_shift_grad(frame, track_window, roi_hist, kernel_size=3):
    # convert to HSV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # histogram back projection using roi_hist 
    dst = cv2.calcBackProject([gray],[0],roi_hist,[0,360],1)
    
    # use meanshift to shift the tracking window
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    return track_window

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
    roi = frame[y:y+h, x:x+w]

    roi_hist = initialize_grad_hist(roi)

    ious = []

    while True:
        # grab a frame
        ret ,frame = cap.read() 
        
        if ret == True: 
            # convert to HSV
            track_window_2 = detect_closest_face(detect_faces(frame), track_window)
            (x2,y2,w2,h2) = track_window_2

            track_window = mean_shift_grad(frame, track_window, roi_hist)
            (x,y,w,h) = track_window

            ious.append(get_intersection_over_union(track_window, track_window_2))

            img = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255) ,5)
            img = cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (255,0,0), 5)
            
            cv2.imshow('mean shift tracking demo', img)
            
            if cv2.waitKey(1) & 0xFF == 27: # wait a bit and exit is ESC is pressed
                break
            
        else:
            break
    
    plot_iou_across_frames(ious)
            
    cv2.destroyAllWindows()
    cap.release()

    