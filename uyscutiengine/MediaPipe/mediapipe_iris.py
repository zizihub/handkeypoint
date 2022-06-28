import mediapipe as mp
import cv2
from time import time
mp_iris = mp.solutions.iris
mp_draw = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)
    with mp_iris.Iris() as iris:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            start = time()
            results = iris.process(image)
            print('>>> cost time: {}'.format(round((time()-start)*1000, 2)))

            # Draw the eye and iris annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.face_landmarks_with_iris:
                mp_draw.draw_iris_landmarks(image, results.face_landmarks_with_iris)
            cv2.imshow('MediaPipe Iris', image)
            if cv2.waitKey(5) == ord('q'):
                break
    cap.release()


if __name__ == '__main__':
    main()
