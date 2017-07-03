import numpy as np
import cv2

# initialize video capture
cap = cv2.VideoCapture(0)
lastFrame = None
thisFrame = None
frameWidth = int(cap.get(3))
frameHeight = int(cap.get(4))

count = 0

# continue video recording
while (True):

    # can quit out if necessary
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    # create a lastFrame
    if lastFrame is None:
        lastFrame = np.zeros((frameHeight,frameWidth,1), np.uint8)
        continue

    # initialize thisFrame
    thisFrame = np.zeros((frameHeight,frameWidth,1), np.uint8)

    # capture frame
    ret, frame = cap.read()
    # flip frame around
    mirror = cv2.flip(frame, 1)

    # convert to grayscale
    gray = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)

    # find contours
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find squares
    for cnt in contours:

        # ignore smaller areas
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue

        # approximate contour as shape
        approx = cv2.approxPolyDP(cnt, .1 * cv2.arcLength(cnt,True), True)

        # check if the shape is a square
        if len(approx) == 4:

            # draw the square on the frame
            cv2.drawContours(gray, [cnt], 0, (0,0,255), -1)
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw the contours on thisFrame
            cv2.drawContours(thisFrame, [cnt], 0, 255, -1)

    # compute change in frames
    frameDelta = cv2.absdiff(thisFrame, lastFrame)

    # find contours in delta
    ret, thresh = cv2.threshold(frameDelta, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # iterate through contours
    for cnt in contours:

        # ignore small changes
        area = cv2.contourArea(cnt)
        if area < 3000:
            continue

        else:
            print(count)
            count += 1


    # show the frameDelta
    cv2.imshow('Delta', frameDelta)

    # show the frame
    cv2.imshow('Gray', gray)

    # update lastFrame
    lastFrame = thisFrame

# release capture on completion
cap.release()
cv2.destroyAllWindows()
