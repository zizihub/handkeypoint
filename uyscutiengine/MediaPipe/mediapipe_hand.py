import cv2
import mediapipe as mp
import math
from time import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class SimpleGestureDetector:
    # region: Member variables
    # mediaPipe configuration hands object
    __mpHands = mp_hands
    # mediaPipe detector objet
    __mpHandDetector = None

    def __init__(self):
        self.__setDefaultHandConfiguration()

    def __setDefaultHandConfiguration(self):
        self.__mpHandDetector = self.__mpHands.Hands(
            # static images
            static_image_mode=True,
            # default = 2
            max_num_hands=2,
            # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully (default= 0.5)
            min_detection_confidence=0.75,
            # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. (default = 0.5)
            min_tracking_confidence=0.75
        )

    def __getEuclideanDistance(self, posA, posB):
        return math.sqrt((posA.x - posB.x)**2 + (posA.y - posB.y)**2)

    def __isThumbNearIndexFinger(self, thumbPos, indexPos):
        return self.__getEuclideanDistance(thumbPos, indexPos) < 0.1

    def detectHands(self, image):
        if self.__mpHandDetector is None:
            return

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # start handDetector on current image
        results = self.__mpHandDetector.process(image)
        # unlock image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(dir(results))
        # print(results.palm_detections)
        if results.multi_hand_landmarks and results.palm_detections:
            for idx, (hand_landmarks, hand_bboxes) in enumerate(zip(results.multi_hand_landmarks, results.palm_detections)):
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # mp_drawing.draw_detection(image, hand_bboxes)
                gesture = self.simpleGesture(hand_landmarks)
                image = self.draw_gesture(image, gesture, idx)
        # Draw hand world landmarks.
        # if not results.multi_hand_world_landmarks:
        #     return image
        # for hand_world_landmarks in results.multi_hand_world_landmarks:
        #     mp_drawing.plot_landmarks(
        #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
        return image

    def draw_gesture(self, image, gesture, idx):
        image = cv2.putText(image, 'Hand {}: {}'.format(idx+1, gesture), (50, 50*(idx+1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, 1)
        return image

    def simpleGesture(self, handLandmarks):

        thumbIsOpen = False
        indexIsOpen = False
        middelIsOpen = False
        ringIsOpen = False
        pinkyIsOpen = False

        pseudoFixKeyPoint = handLandmarks.landmark[2].x
        if handLandmarks.landmark[3].x < pseudoFixKeyPoint and handLandmarks.landmark[4].x < pseudoFixKeyPoint:
            thumbIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[6].y
        if handLandmarks.landmark[7].y < pseudoFixKeyPoint and handLandmarks.landmark[8].y < pseudoFixKeyPoint:
            indexIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[10].y
        if handLandmarks.landmark[11].y < pseudoFixKeyPoint and handLandmarks.landmark[12].y < pseudoFixKeyPoint:
            middelIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[14].y
        if handLandmarks.landmark[15].y < pseudoFixKeyPoint and handLandmarks.landmark[16].y < pseudoFixKeyPoint:
            ringIsOpen = True

        pseudoFixKeyPoint = handLandmarks.landmark[18].y
        if handLandmarks.landmark[19].y < pseudoFixKeyPoint and handLandmarks.landmark[20].y < pseudoFixKeyPoint:
            pinkyIsOpen = True

        if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FIVE!")
            return "FIVE!"

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
            print("FOUR!")
            return "FOUR!"

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
            print("THREE!")
            return "THREE!"

        elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("TWO!")
            return "TWO!"

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("ONE!")
            return "ONE!"

        elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("ROCK!")
            return "ROCK!"

        elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen and self.__isThumbNearIndexFinger(handLandmarks.landmark[3], handLandmarks.landmark[7]):
            print("HEART!")
            return "HEART!"

        elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
            print("SPIDERMAN!")
            return "SPIDERMAN!"

        elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
            print("FIST!")
            return "FIST!"

        elif not indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen and self.__isThumbNearIndexFinger(handLandmarks.landmark[4], handLandmarks.landmark[8]):
            print("OK!")
            return "OK!"

        print("FingerState: thumbIsOpen? " + str(thumbIsOpen) + " - indexIsOpen? " + str(indexIsOpen) + " - middelIsOpen? " +
              str(middelIsOpen) + " - ringIsOpen? " + str(ringIsOpen) + " - pinkyIsOpen? " + str(pinkyIsOpen))


def hand_tracking():
    import os
    if 0:
        root = '../MNN Engine/datasets/gesture_new'
        file_list = os.listdir(root)
        file_list = list(map(lambda x: os.path.join(root, x), file_list))
        # For static images:
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(file_list):
                # Read an image, flip it around y-axis for correct handedness output (see
                # above).
                image = cv2.flip(cv2.imread(file), 1)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print handedness and draw hand landmarks on the image.
                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    print('hand_landmarks:', hand_landmarks)
                    print(
                        f'Index finger tip coordinates: (',
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    )
                    mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imwrite(
                    './tmp/annotated_image' + str(idx) + '.png', annotated_image)
                # Draw hand world landmarks.
                if not results.multi_hand_world_landmarks:
                    continue
                for hand_world_landmarks in results.multi_hand_world_landmarks:
                    mp_drawing.plot_landmarks(
                        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    else:
        # For webca`m input:
        cap = cv2.VideoCapture(0)
        sgd = SimpleGestureDetector()
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            start = time()
            image = sgd.detectHands(image)
            print('>>> cost time: {}'.format(round((time()-start)*1000, 2)))

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()


if __name__ == '__main__':
    hand_tracking()
