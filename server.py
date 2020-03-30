from imutils import build_montages
from datetime import datetime
import imagezmq
import imutils
import cv2


imageHub = imagezmq.ImageHub()
frameDict = {}
lastActive = {}
lastActiveCheck = datetime.now()
ESTIMATED_NUM_PIS = 4
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

while True:
    (rpiName, frame) = imageHub.recv_image()
    imageHub.send_reply(b'OK')

    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))

    lastActive[rpiName] = datetime.now()

    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    frameDict[rpiName] = frame

    montages = build_montages(frameDict.values(), (w, h), (2, 2))

    for (i, montage) in enumerate(montages):
        cv2.imshow("Home pet location monitor ({})".format(i), montage)

    key = cv2.waitKey(1) & 0xFF

    if (datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        for (rpiName, ts) in list(lastActive.items()):
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                lastActive.pop(rpiName)
                frameDict.pop(rpiName)

        lastActiveCheck = datetime.now()

    if key == ord("q"):
        break

cv2.destroyAllWindows()