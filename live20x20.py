from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import import_intensity
import numpy as np

import_intensity.preprocess_data(20, 20)
knn = import_intensity.trainKnn()

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
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    cv2.rectangle(frame, (180, 180), (220, 220),
                  233, 2)

    digit = frame[180:220, 180:220]
    digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_CUBIC)
    img = import_intensity.preprocess_test(img)
    cv2.imshow("Cropped", img)
    img = np.array(img).reshape(-1, 400).astype(np.float32)
    result = import_intensity.doKnn(knn, img, 3)

    cv2.putText(frame, str(i) + str(result), (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 0, 1)

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
