from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import import_intensity
import numpy as np

import_intensity.preprocess_data(34, 40)
knn_i = import_intensity.trainKnn()

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
i = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = cv2.flip(vs.read(), 1)
    # frame = vs.read()
    frame = imutils.resize(frame, width=600)

    cv2.rectangle(frame, (30, 30), (150, 50), (255, 255, 255), cv2.FILLED)

    digit = frame[133:167, 130:170]
    digit = cv2.flip(digit, 1)
    digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    digit = cv2.resize(digit, (34, 40), interpolation=cv2.INTER_CUBIC)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(digit, kernel, iterations=1)
    # img = cv2.bitwise_not(img)
    thres = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(thres)

    r_img = np.array(img).reshape(-1, 34*40).astype(np.float32)
    stack = np.hstack((digit, img, thres))
    cv2.imshow("digit, img, thres", stack)

    result_intensity = import_intensity.doKnn(knn_i, r_img, 5)
    cv2.putText(frame, str(i) + " i: " + str(result_intensity),
                (40, 44), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 0), 1)

    cv2.rectangle(frame, (133, 130), (160, 170), 233, 2)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    i += 1
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
