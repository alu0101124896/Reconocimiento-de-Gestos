from math import atan2

import cv2 as cv
import numpy as np

LOW_RESOLUTION_WEBCAM = True


def main():
    """Main function of the program."""

    # Capture the video from the default camera
    default_camera = 0
    captured_video = cv.VideoCapture(default_camera)

    # Initialization of the background subtractor
    background_subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=True)

    if not captured_video.isOpened():
        print("Error: Unable to open file")
        exit(0)

    # Top-left and bottom-right points of the region of interest rectangle:

    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        region_of_interest_point1 = (330, 10)
        region_of_interest_point2 = (630, 310)
    else:
        # Medium resolution webcam
        region_of_interest_point1 = (800, 30)
        region_of_interest_point2 = (1250, 530)

    # Constant tuple with the two learning rates for the background subtractor
    learning_rates = (0.3, 0)

    # Initialization of the current learning rate
    current_learning_rate = learning_rates[0]

    # Boolean that stores if the user wants to count the raised fingers or not
    count_hand_fingers = True

    # Boolean that stores if the user wants to detect hand gestures or not
    detect_hand_gestures = True

    # Boolean that stores if the user wants to draw with the index finger or not
    index_finger_drawing = False

    # List that stores the trace of the current stroke of the drawing
    current_stroke = list()

    # List that stores the entire drawing
    current_drawing = list()

    # Boolean that stores if the user wants to see the help info or not
    show_help = True

    while True:
        # Read the data from the captured video
        success, captured_frame = captured_video.read()
        if not success:
            print("Error: Unable to get data")
            exit(0)

        # Window showing the mirrored captured video and the region of interest marked
        #  with a blue rectangle
        captured_frame = cv.flip(captured_frame, 1)
        cv.rectangle(
            captured_frame,
            region_of_interest_point1,
            region_of_interest_point2,
            color=(255, 0, 0),
        )
        cv.imshow("WebCam", captured_frame)

        # Window showing the region of interest only
        region_of_interest = captured_frame[
            region_of_interest_point1[1] : region_of_interest_point2[1],
            region_of_interest_point1[0] : region_of_interest_point2[0],
            :,
        ].copy()
        # cv.imshow('Region of Interest', region_of_interest)

        # Window showing the background subtraction applied
        foreground_mask = background_subtractor.apply(
            region_of_interest,
            None,
            current_learning_rate,
        )
        # cv.imshow('Foreground Mask', foreground_mask)

        # Window showing the gray threshold applied
        success, black_and_white = cv.threshold(
            foreground_mask,
            200,
            255,
            cv.THRESH_BINARY,
        )
        cv.imshow("Black and White", black_and_white)

        # Window showing the hand contour
        contours = cv.findContours(
            black_and_white,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )[0]
        contour_window = region_of_interest.copy()

        if len(contours) > 0 and current_learning_rate != learning_rates[0]:
            hand_contour = get_largest_contour(contours)
            cv.drawContours(
                contour_window,
                hand_contour,
                contourIdx=-1,
                color=(0, 255, 0),
                thickness=3,
            )
        else:
            hand_contour = None

        cv.imshow("Contour", contour_window)

        # Window showing the hand's convex hull
        convex_hull_window = region_of_interest.copy()

        if hand_contour is not None:
            hand_convex_hull = cv.convexHull(hand_contour)
            cv.drawContours(
                convex_hull_window,
                [hand_convex_hull],
                contourIdx=0,
                color=(255, 0, 0),
                thickness=3,
            )
        else:
            hand_convex_hull = None

        cv.imshow("Convex Hull", convex_hull_window)

        # Window showing the fingers' convexity defects
        convexity_defects_window = region_of_interest.copy()

        if hand_contour is not None:
            hand_convex_hull = cv.convexHull(
                hand_contour,
                clockwise=False,
                returnPoints=False,
            )
            temp_python_list = list(hand_convex_hull)
            temp_python_list.sort(reverse=True, key=lambda element: element[0])
            hand_convex_hull = np.array(temp_python_list)
            hand_convexity_defects = cv.convexityDefects(hand_contour, hand_convex_hull)
            finger_convexity_defects = list()

            if hand_convexity_defects is not None:
                for current_convexity_defect in range(len(hand_convexity_defects)):
                    (
                        start_index,
                        end_index,
                        far_index,
                        distance_to_convex_hull,
                    ) = hand_convexity_defects[current_convexity_defect][0]

                    start_point = tuple(hand_contour[start_index][0])
                    end_point = tuple(hand_contour[end_index][0])
                    far_point = tuple(hand_contour[far_index][0])

                    depth = distance_to_convex_hull / 256.0

                    if depth > 80.0:
                        # angle_of_current_convexity_defect = angle(
                        #     start_point,
                        #     end_point,
                        #     far_point,
                        # )
                        cv.line(
                            convexity_defects_window,
                            start_point,
                            end_point,
                            color=(255, 0, 0),
                            thickness=2,
                        )
                        cv.circle(
                            convexity_defects_window,
                            far_point,
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1,
                        )
                        finger_convexity_defects.append(
                            (start_point, end_point, far_point)
                        )

        else:
            hand_convex_hull = None
            hand_convexity_defects = None

        cv.imshow("Convexity Defects", convexity_defects_window)

        # Window showing the hand's bounding rectangle
        bounding_rectangle_window = region_of_interest.copy()

        if hand_contour is not None:
            hand_bounding_rectangle = cv.boundingRect(hand_contour)

            bounding_rectangle_point1 = (
                hand_bounding_rectangle[0],
                hand_bounding_rectangle[1],
            )
            bounding_rectangle_point2 = (
                hand_bounding_rectangle[0] + hand_bounding_rectangle[2],
                hand_bounding_rectangle[1] + hand_bounding_rectangle[3],
            )

            cv.rectangle(
                bounding_rectangle_window,
                bounding_rectangle_point1,
                bounding_rectangle_point2,
                color=(0, 0, 255),
                thickness=3,
            )
        else:
            hand_bounding_rectangle = None
            bounding_rectangle_point1 = None
            bounding_rectangle_point2 = None

        cv.imshow("Bounding Rectangle", bounding_rectangle_window)

        # Window showing the user side functionalities
        main_window = region_of_interest.copy()

        if hand_contour is not None:
            number_of_fingers = count_fingers(
                finger_convexity_defects,
                hand_bounding_rectangle,
            )

            if count_hand_fingers:
                main_window = print_fingers(number_of_fingers, main_window)

            if detect_hand_gestures:
                hand_gesture = detect_hand_gesture(
                    number_of_fingers,
                    finger_convexity_defects,
                    hand_bounding_rectangle,
                )
                main_window = print_gestures(hand_gesture, main_window)

            if index_finger_drawing:
                current_stroke = finger_drawing(current_stroke, hand_contour)

            main_window = print_stroke(current_stroke, main_window)

            for stroke in current_drawing:
                main_window = print_stroke(stroke, main_window)

        if show_help:
            main_window = print_help(main_window)

        if current_learning_rate == learning_rates[0]:
            main_window = print_learning(main_window)

        cv.imshow("Main Window", main_window)

        keyboard = cv.waitKey(1)

        # Key used for swapping between the two learning rates
        if keyboard & 0xFF == ord("s"):
            current_learning_rate = swap_learning_rate(
                current_learning_rate,
                learning_rates,
            )

        # Key used for swapping between counting the raised fingers or not
        if keyboard & 0xFF == ord("f"):
            count_hand_fingers = not count_hand_fingers

        # Key used for swapping between detecting hand gestures or not
        if keyboard & 0xFF == ord("g"):
            detect_hand_gestures = not detect_hand_gestures

        # Key used for swapping between drawing with the index finger or not
        if keyboard & 0xFF == ord("d"):
            index_finger_drawing = not index_finger_drawing

            if not index_finger_drawing:
                current_drawing.append(current_stroke[:])
                current_stroke.clear()

        # Key used for cleaning the last stroke
        if keyboard & 0xFF == ord("c"):
            current_drawing.pop()

        # Key used for cleaning the entire drawing
        if keyboard & 0xFF == ord("x"):
            current_drawing.clear()

        # Key used for swapping between showing the help info or not
        if keyboard & 0xFF == ord("h"):
            show_help = not show_help

        # Key used for finishing the program execution
        if keyboard & 0xFF == ord("q"):
            break

    captured_video.release()
    cv.destroyAllWindows()


def get_largest_contour(contours):
    """Function that returns the largest contour. In this case it's the hand contour."""

    largest_contour = contours[0]
    size_of_largest_contour = contours[0].size

    for current_contour in contours:
        size_of_current_contour = current_contour.size

        if size_of_current_contour > size_of_largest_contour:
            largest_contour = current_contour
            size_of_largest_contour = size_of_current_contour

    return largest_contour


def count_fingers(finger_convexity_defects, hand_bounding_rectangle):
    """Function that counts the number of raised fingers."""

    number_of_fingers = len(finger_convexity_defects)

    if number_of_fingers > 0:
        return number_of_fingers + 1
    elif fingers_raised(hand_bounding_rectangle):
        return 1
    else:
        return 0


def fingers_raised(hand_bounding_rectangle):
    """Function that checks if any finger is raised."""

    if hand_bounding_rectangle[3] > (hand_bounding_rectangle[2] * 1.3):
        return True
    else:
        return False


def detect_hand_gesture(
    number_of_fingers,
    finger_convexity_defects,
    hand_bounding_rectangle,
):
    """Function that checks what gesture is being made with the hand."""

    if hand_bounding_rectangle[2] > 25 and hand_bounding_rectangle[3] > 25:
        if number_of_fingers == 0:
            return "Raised Fist"

        elif number_of_fingers == 2:
            first_finger, second_finger, mid_point = finger_convexity_defects[0]

            if angle(first_finger, second_finger, mid_point) <= 60:
                return "Peace"

            elif 60 < angle(first_finger, second_finger, mid_point) <= 90:
                return "Rock'n'Roll"

            elif 90 < angle(first_finger, second_finger, mid_point):
                return "Shaka"

        elif number_of_fingers == 4:
            return "Ok"

        else:
            return "Not recognised"

    else:
        return "Not recognised"


def angle(start_point, end_point, far_point):
    """Function that calculates the angle of the three given points."""

    vector1 = [
        start_point[0] - far_point[0],
        start_point[1] - far_point[1],
    ]
    vector2 = [
        end_point[0] - far_point[0],
        end_point[1] - far_point[1],
    ]

    angle1 = atan2(vector1[1], vector1[0])
    angle2 = atan2(vector2[1], vector2[0])
    angle3 = angle1 - angle2

    if angle3 > np.pi:
        angle3 -= 2 * np.pi
    if angle3 < -np.pi:
        angle3 += 2 * np.pi

    return angle3 * 180 / np.pi


def finger_drawing(stroke, hand_contour):
    """Function that adds the current index position to the drawing."""

    stroke.append(tuple(hand_contour[0][0]))
    return stroke


def print_fingers(number_of_fingers, window):
    """Function that prints the number of fingers raised in the given window."""

    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 280)
    else:
        # Medium resolution webcam
        position = (10, 480)

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2
    line_type = cv.LINE_AA

    cv.putText(
        window,
        text=f"Fingers: {number_of_fingers}",
        org=position,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )

    return window


def print_gestures(hand_gesture, window):
    """Function that prints the name of the hand gesture on the given window."""

    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 250)
    else:
        # Medium resolution webcam
        position = (10, 450)

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2
    line_type = cv.LINE_AA

    cv.putText(
        window,
        text=f"Gesture: {hand_gesture}",
        org=position,
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )

    return window


def print_stroke(drawing, window):
    """Function that prints the finger drawing on the given window."""

    for point_index in range(1, len(drawing)):
        cv.line(
            window,
            drawing[point_index - 1],
            drawing[point_index],
            color=(0, 0, 255),
            thickness=2,
        )

    return window


def print_learning(window):
    """Function that prints a learning message in the given window."""

    if LOW_RESOLUTION_WEBCAM:
        # Low resolution webcam
        position = (10, 280)
    else:
        # Medium resolution webcam
        position = (10, 480)

    cv.putText(
        window,
        text="Learning...",
        org=position,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv.LINE_AA,
    )

    return window


def print_help(window):
    """Function that prints the help info in the given window."""

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255)
    thickness = 1
    line_type = cv.LINE_AA

    cv.putText(
        window,
        text="s: start/stop learning background",
        org=(10, 20),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="f: count number of fingers",
        org=(10, 40),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="g: detect hand gesture",
        org=(10, 60),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="d: draw with one finger",
        org=(10, 80),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="c: clean last stroke",
        org=(10, 100),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="x: clean drawing",
        org=(10, 120),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="h: show/hide help",
        org=(10, 140),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )
    cv.putText(
        window,
        text="q: quit program",
        org=(10, 160),
        fontFace=font_face,
        fontScale=font_scale,
        color=color,
        thickness=thickness,
        lineType=line_type,
    )

    return window


def swap_learning_rate(current_learning_rate, learning_rates):
    """Function that swaps between the two learning rates stored on a global constant
    tuple defined at the beginning of this script file."""

    if current_learning_rate == learning_rates[0]:
        current_learning_rate = learning_rates[1]
    else:
        current_learning_rate = learning_rates[0]

    return current_learning_rate


main()
