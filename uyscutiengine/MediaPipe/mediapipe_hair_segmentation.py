import mediapipe as mp
import os
import cv2
import numpy as np
from time import time
from PIL import Image
from datetime import timedelta

mp_hair_segmentation = mp.solutions.hair_segmentation
mp_draw = mp.solutions.drawing_utils
drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)


def cam_inference(video_path=0, save_video=False):
    # For webcam input:
    # save video
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter('/Users/markson/WorkSpace/uyscutiengine/MNN Engine/datasets/video/hair.mp4',
                              fourcc, 15.0, (720, 1280), True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    with mp_hair_segmentation.HairSegmentation() as hair_segmentation:
        while cap.isOpened():
            success, image = cap.read()
            count += 1
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            start = time()
            results = hair_segmentation.process(image)
            print('>>> [{}] cost time: {}'.format(count, round((time()-start)*1000, 2)))
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # channel 1 == 4, channel 2 == 3
            alpha = results.hair_mask[:, :, -1] / 255.
            alpha = cv2.resize(alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            alpha = cv2.merge((alpha, alpha, alpha))
            cv2.imshow('Alpha Mat', alpha)

            # output video
            out_video = results.output_video
            # cv2.imshow('output video', cv2.cvtColor(out_video, cv2.COLOR_RGB2BGR))

            # recolor
            color = [106, 109, 208]
            color_map = np.ones(image.shape[:2])
            color_map = cv2.merge((color_map*color[0], color_map*color[1], color_map*color[2]))
            bg = cv2.multiply(alpha, color_map, dtype=cv2.CV_8UC3)
            vis_img = cv2.addWeighted(image, 1, bg, 0.5, 0)

            print('vis image shape', vis_img.shape)
            # cv2.imshow('Original', image)
            cv2.imshow('MediaPipe Hair Segmentation', vis_img)
            if save_video:
                out.write(alpha)

            # output video
            # output_video = results.output_video
            # output_video = cv2.cvtColor(output_video, cv2.COLOR_RGB2BGR)
            # cv2.imshow('MediaPipe Video', output_video)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    if save_video:
        out.release()


def img_inference(img_folder):
    out_dir = '/Volumes/Lexar/data/face_data/CelebA-HQ/CelebA-HQ-AlphaMat'
    os.makedirs(out_dir, exist_ok=True)
    hair_segmentation = mp_hair_segmentation.HairSegmentation()
    for root, dirs, files in os.walk(img_folder):
        if dirs:
            continue
        tot = len(files)
        cost = []
        eta = None
        for i, f in enumerate(files):
            if cost:
                eta = timedelta(seconds=np.nanmean(cost)) * (tot - i)
            print('[%5d/%d] processing  %10s  |  eta: %s' % (i+1, tot, f, eta))
            dst = os.path.join(out_dir, f.replace('jpg', 'png'))
            if os.path.exists(dst):
                print('file exists, continue...')
                continue

            start = time()
            image = cv2.imread(os.path.join(img_folder, f))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            count = 0
            results = hair_segmentation.process(image)
            alpha_list = results.hair_mask[:, :, -1]
            while count < 20:
                count += 1
                results = hair_segmentation.process(image)
                alpha = results.hair_mask[:, :, -1]
                if count % 5 != 0:
                    continue
                if isinstance(alpha_list, np.ndarray):
                    alpha_list = np.hstack([alpha_list, alpha])
                else:
                    alpha_list = alpha
            alpha_list = alpha_list.astype(np.uint8)
            # alpha_list = cv2.cvtColor(alpha_list, cv2.COLOR_GRAY2BGR)
            # cv2.imshow('Alpha Mat', np.hstack(
            #     [cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512, 512)), alpha_list]))
            cv2.imwrite(dst, alpha)
            cost.append(time()-start)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    cam_inference()
