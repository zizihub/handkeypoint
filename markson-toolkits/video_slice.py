import cv2
import os
from multiprocessing import Pool
from moviepy.editor import VideoFileClip


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break


def video_slice(video_path):
    print('processing {}...'.format(video_path))
    video_name = video_path.split('/')[-1].split('.')[0]
    dst = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(dst, exist_ok=True)
    for i, frame in enumerate(get_frames(video_path)):
        # if i % 20 != 0:
        #     continue
        cv2.imwrite('{}/{}_{}.jpg'.format(dst, video_name, i), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def video_convert(video_path):
    print('processing {}...'.format(video_path))
    video_name = video_path.split('/')[-1].split('.')[0]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = None
    for frame in get_frames(video_path):
        if out is None:
            out = cv2.VideoWriter('/Users/markson/WorkSpace/pysot/result/{}.mp4'.format(video_name), fourcc,
                                  30.0, (frame.shape[1], frame.shape[0]), True)
        cv2.imshow(video_name, frame)
        cv2.waitKey(1)
        out.write(frame)
    cv2.destroyAllWindows()
    out.release()


def mp42gif(video_path):
    assert video_path.endswith(('.mp4', '.MP4'))
    print('processing {}...'.format(video_path))
    video_name = video_path.split('/')[-1].split('.')[0]
    clip = (VideoFileClip(video_path).resize(0.75))
    clip.write_gif(
        '/Users/markson/WorkSpace/pysot/result/{}.gif'.format(video_name))


if __name__ == '__main__':
    video_slice('../uyscutiengine/MNN Engine/datasets/video/gesture_new.mp4')
