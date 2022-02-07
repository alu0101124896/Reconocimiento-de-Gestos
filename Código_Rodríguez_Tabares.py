import numpy
import math

import cv2

LOW_RESOLUTION_WEBCAM = True


def main():
    """
    Main function of the program.
    """
    # Capture the video from the default camera
    defaultCamera = 0
    capturedVideo = cv2.VideoCapture(defaultCamera)

    # Initialization of the background subtractor
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True)

    if not capturedVideo.isOpened():
        print('Error: Unable to open file')
        exit(0)

    # Top-left and bottom-right points of the region of interest rectangle:

    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        regionOfInterestPoint1 = (330, 10)
        regionOfInterestPoint2 = (630, 310)
    else:
        # Medium resolution webcam
        regionOfInterestPoint1 = (800, 30)
        regionOfInterestPoint2 = (1250, 530)

    # Constant tuple with the two learning rates for the background
    #  subtractor
    learningRates = (0.3, 0)

    # Initialization of the current learning rate
    currentLearningRate = learningRates[0]

    # Boolean that stores if the user wants to count the raised fingers or not
    countHandFingers = True

    # Boolean that stores if the user wants to detect hand gestures or not
    detectHandGestures = True

    # Boolean that stores if the user wants to draw with the index finger or not
    indexFingerDrawing = False

    # List that stores the trace of the current stroke of the drawing
    currentStroke = list()

    # List that stores the entire drawing
    currentDrawing = list()

    # Boolean that stores if the user wants to see the help info or not
    showHelp = True

    while True:
        # Read the data from the captured video
        returntErrorValue, capturedFrame = capturedVideo.read()
        if not returntErrorValue:
            print('Error: Unable to get data')
            exit(0)

        # Window showing the mirrored captured video and the region of interest
        #  marked with a blue rectangle
        capturedFrame = cv2.flip(capturedFrame, 1)
        cv2.rectangle(capturedFrame,
                      regionOfInterestPoint1,
                      regionOfInterestPoint2,
                      color=(255, 0, 0))
        cv2.imshow('WebCam', capturedFrame)

        # Window showing the region of interest only
        regionOfInterest = capturedFrame[
            regionOfInterestPoint1[1]:regionOfInterestPoint2[1],
            regionOfInterestPoint1[0]:regionOfInterestPoint2[0], :].copy()
        # cv2.imshow('Region of Interest', regionOfInterest)

        # Window showing the background subtraction applied
        foregroundMask = backgroundSubtractor.apply(regionOfInterest, None,
                                                    currentLearningRate)
        # cv2.imshow('Foreground Mask', foregroundMask)

        # Window showing the gray threshold applied
        returntErrorValue, blackAndWhite = cv2.threshold(
            foregroundMask, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('Black and White', blackAndWhite)

        # Window showing the hand contour
        contours = cv2.findContours(blackAndWhite, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        contourWindow = regionOfInterest.copy()

        if len(contours) > 0 and currentLearningRate != learningRates[0]:
            handContour = getLargerContour(contours)
            cv2.drawContours(contourWindow,
                             handContour,
                             contourIdx=-1,
                             color=(0, 255, 0),
                             thickness=3)
        else:
            handContour = None

        cv2.imshow('Contour', contourWindow)

        # Window showing the hand's convex hull
        convexHullWindow = regionOfInterest.copy()

        if handContour is not None:
            handConvexHull = cv2.convexHull(handContour)
            cv2.drawContours(convexHullWindow, [handConvexHull],
                             contourIdx=0,
                             color=(255, 0, 0),
                             thickness=3)
        else:
            handConvexHull = None

        cv2.imshow('Convex Hull', convexHullWindow)

        # Window showing the fingers' convexity defects
        convexityDefectsWindow = regionOfInterest.copy()

        if handContour is not None:
            handConvexHull = cv2.convexHull(handContour,
                                            clockwise=False,
                                            returnPoints=False)
            tempPythonList = list(handConvexHull)
            tempPythonList.sort(reverse=True, key=lambda element: element[0])
            handConvexHull = numpy.array(tempPythonList)
            handConvexityDefects = cv2.convexityDefects(
                handContour, handConvexHull)
            fingerConvexityDefects = list()

            if handConvexityDefects is not None:
                for currentConvexityDefect in range(len(handConvexityDefects)):
                    startIndex, endIndex, farIndex, distanceToConvexHull \
                        = handConvexityDefects[currentConvexityDefect][0]

                    startPoint = tuple(handContour[startIndex][0])
                    endPoint = tuple(handContour[endIndex][0])
                    farPoint = tuple(handContour[farIndex][0])

                    depth = distanceToConvexHull / 256.0

                    if depth > 80.0:
                        # angleOfCurrentConvexityDefect = angle(
                        #     startPoint, endPoint, farPoint)
                        cv2.line(convexityDefectsWindow,
                                 startPoint,
                                 endPoint,
                                 color=(255, 0, 0),
                                 thickness=2)
                        cv2.circle(convexityDefectsWindow,
                                   farPoint,
                                   radius=5,
                                   color=(0, 0, 255),
                                   thickness=-1)
                        fingerConvexityDefects.append(
                            (startPoint, endPoint, farPoint))

        else:
            handConvexHull = None
            handConvexityDefects = None

        cv2.imshow('Convexity Defects', convexityDefectsWindow)

        # Window showing the hand's bounding rectangle
        boundingRectangleWindow = regionOfInterest.copy()

        if handContour is not None:
            handBoundingRectangle = cv2.boundingRect(handContour)

            boundingRectanglePoint1 = (handBoundingRectangle[0],
                                       handBoundingRectangle[1])
            boundingRectanglePoint2 = (handBoundingRectangle[0] +
                                       handBoundingRectangle[2],
                                       handBoundingRectangle[1] +
                                       handBoundingRectangle[3])

            cv2.rectangle(boundingRectangleWindow,
                          boundingRectanglePoint1,
                          boundingRectanglePoint2,
                          color=(0, 0, 255),
                          thickness=3)
        else:
            handBoundingRectangle = None
            boundingRectanglePoint1 = None
            boundingRectanglePoint2 = None

        cv2.imshow('Bounding Rectangle', boundingRectangleWindow)

        # Window showing the user side functionalities
        mainWindow = regionOfInterest.copy()

        if handContour is not None:
            numberOfFingers = countFingers(fingerConvexityDefects,
                                           handBoundingRectangle)

            if countHandFingers:
                mainWindow = printFingers(numberOfFingers, mainWindow)

            if detectHandGestures:
                handGesture = detectHandGesture(numberOfFingers,
                                                fingerConvexityDefects,
                                                handBoundingRectangle)
                mainWindow = printGestures(handGesture, mainWindow)

            if indexFingerDrawing:
                currentStroke = fingerDrawing(currentStroke, handContour,
                                              mainWindow)

            mainWindow = printStroke(currentStroke, mainWindow)

            for stroke in currentDrawing:
                mainWindow = printStroke(stroke, mainWindow)

        if showHelp:
            mainWindow = printHelp(mainWindow)

        if currentLearningRate == learningRates[0]:
            mainWindow = printLearning(mainWindow)

        cv2.imshow('Main Window', mainWindow)

        keyboard = cv2.waitKey(1)

        # Key used for swaping between the two learning rates
        if keyboard & 0xFF == ord('s'):
            currentLearningRate = swapLearningRate(currentLearningRate,
                                                   learningRates)

        # Key used for swaping between counting the raised fingers or not
        if keyboard & 0xFF == ord('f'):
            countHandFingers = not countHandFingers

        # Key used for swaping between detecting hand gestures or not
        if keyboard & 0xFF == ord('g'):
            detectHandGestures = not detectHandGestures

        # Key used for swaping between drawing with the index finger or not
        if keyboard & 0xFF == ord('d'):
            indexFingerDrawing = not indexFingerDrawing

            if not indexFingerDrawing:
                currentDrawing.append(currentStroke[:])
                currentStroke.clear()

        # Key used for cleaning the last stroke
        if keyboard & 0xFF == ord('c'):
            currentDrawing.pop()

        # Key used for cleaning the entire drawing
        if keyboard & 0xFF == ord('x'):
            currentDrawing.clear()

        # Key used for swaping between showing the help info or not
        if keyboard & 0xFF == ord('h'):
            showHelp = not showHelp

        # Key used for finishing the program execution
        if keyboard & 0xFF == ord('q'):
            break

    capturedVideo.release()
    cv2.destroyAllWindows()


def getLargerContour(contours):
    """
    Function that returns the larger contour.
    In this case it's the hand contour.
    """
    largerContour = contours[0]
    sizeOfLargerContour = contours[0].size

    for currentContour in contours:
        sizeOfCurrentContour = currentContour.size

        if sizeOfCurrentContour > sizeOfLargerContour:
            largerContour = currentContour
            sizeOfLargerContour = sizeOfCurrentContour

    return largerContour


def countFingers(fingerConvexityDefects, handBoundingRectangle):
    """
    Function that counts the number of raised fingers.
    """
    numberOfFingers = len(fingerConvexityDefects)

    if numberOfFingers > 0:
        return numberOfFingers + 1
    elif fingersRaised(handBoundingRectangle):
        return 1
    else:
        return 0


def fingersRaised(handBoundingRectangle):
    """
    Function that checks if any finger is raised.
    """
    if handBoundingRectangle[3] > handBoundingRectangle[2] * 1.3:
        return True
    else:
        return False


def detectHandGesture(numberOfFingers, fingerConvexityDefects,
                      handBoundingRectangle):
    """
    Function that checks what gesture is being made with the hand
    """
    if handBoundingRectangle[2] > 25 and handBoundingRectangle[3] > 25:
        if numberOfFingers == 0:
            return 'Raised Fist'

        elif numberOfFingers == 2:
            firstFinger, secondFinger, midPoint = fingerConvexityDefects[0]

            if angle(firstFinger, secondFinger, midPoint) < 60:
                return 'Peace'

            elif 60 < angle(firstFinger, secondFinger, midPoint) < 90:
                return "Rock'n'Roll"

            elif 90 < angle(firstFinger, secondFinger, midPoint):
                return 'Shaka'

        elif numberOfFingers == 4:
            return 'Ok'

        else:
            return 'Not recognised'

    else:
        return 'Not recognised'


def angle(startPoint, endPoint, farPoint):
    """
    Function that calculates the angle of the three given points.
    """
    vector1 = [startPoint[0] - farPoint[0], startPoint[1] - farPoint[1]]
    vector2 = [endPoint[0] - farPoint[0], endPoint[1] - farPoint[1]]

    angle1 = math.atan2(vector1[1], vector1[0])
    angle2 = math.atan2(vector2[1], vector2[0])
    angle3 = angle1 - angle2

    if (angle3 > numpy.pi):
        angle3 -= 2 * numpy.pi
    if (angle3 < -numpy.pi):
        angle3 += 2 * numpy.pi

    return angle3 * 180 / numpy.pi


def fingerDrawing(stroke, handContour, window):
    """
    Function that adds the current index position to the drawing
    """
    stroke.append(tuple(handContour[0][0]))
    return stroke


def printFingers(numberOfFingers, window):
    """
    Function that prints the number of fingers raised in the given window
    """
    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 280)
    else:
        # Medium resolution webcam
        position = (10, 480)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(window,
                text=f'Fingers: {numberOfFingers}',
                org=position,
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)

    return window


def printGestures(handGesture, window):
    """
    Function that prints the name of the hand gesture on the given window
    """
    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 250)
    else:
        # Medium resolution webcam
        position = (10, 450)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(window,
                text=f'Gesture: {handGesture}',
                org=position,
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)

    return window


def printStroke(drawing, window):
    """
    Function that prints the finger drawing on the given window
    """
    for pointIndex in range(1, len(drawing)):
        cv2.line(window,
                 drawing[pointIndex - 1],
                 drawing[pointIndex],
                 color=(0, 0, 255),
                 thickness=2)

    return window


def printLearning(window):
    """
    Function that prints a learning message in the given window
    """
    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 280)
    else:
        # Medium resolution webcam
        position = (10, 480)

    cv2.putText(window,
                text='Learning...',
                org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA)

    return window


def printHelp(window):
    """
    Function that prints the help info in the given window.
    """
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 0, 255)
    thickness = 1
    lineType = cv2.LINE_AA

    cv2.putText(window,
                text='s: start/stop learning background',
                org=(10, 20),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='f: count number of fingers',
                org=(10, 40),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='g: detect hand gesture',
                org=(10, 60),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='d: draw with one finger',
                org=(10, 80),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='c: clean last stroke',
                org=(10, 100),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='x: clean drawing',
                org=(10, 120),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='h: show/hide help',
                org=(10, 140),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)
    cv2.putText(window,
                text='q: quit program',
                org=(10, 160),
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=lineType)

    return window


def swapLearningRate(currentLearningRate, learningRates):
    """
    Function that swaps between the two learning rates stored on a global
    constant tuple defined at the begining of this script file.
    """
    if currentLearningRate == learningRates[0]:
        currentLearningRate = learningRates[1]
    else:
        currentLearningRate = learningRates[0]

    return currentLearningRate


main()
