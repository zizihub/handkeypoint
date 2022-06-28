import random
import string
import numpy as np
import cv2
import math
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.arrayprint import printoptions
# ---------------------------------------------------------------
#                     Segmentation Utils
# ---------------------------------------------------------------


def mask_remove_padding(image, parsing):
    '''
    mask remove zero padding
    '''
    h, w, _ = image.shape
    parsing = cv2.resize(parsing, (max(h, w), max(h, w)), interpolation=cv2.INTER_NEAREST)
    ll = (h - w) // 2
    if ll > 0:
        return cv2.resize(parsing[:, ll:-ll], (w, h), interpolation=cv2.INTER_NEAREST)
    elif ll < 0:
        ll = -ll
        return cv2.resize(parsing[ll:-ll, :], (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        return cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)


def recover_mask(frame, outs, remove_padding=True):
    '''
    recover mask to entire image
    '''
    temp_parsing = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for out, box in outs:
        # * remove padding
        if remove_padding:
            parsing = mask_remove_padding(np.zeros((int(box[3])-int(box[1]), int(box[2])-int(box[0]), 3)), out)
        # * box recover
        temp_parsing[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = parsing
    return temp_parsing


def draw_mask(frame, parsing, show_image=False):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0], [255, 85, 0], [105, 128, 112],
        [85, 96, 225], [255, 0, 170],
        [0, 255, 0], [85, 0, 255], [170, 255, 0],
        [255, 255, 255], [0, 255, 170],
        [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [0, 85, 255], [0, 170, 255],
        [255, 255, 0], [255, 255, 85], [255, 255, 170],
        [255, 0, 255], [255, 85, 255], [255, 170, 255],
        [0, 255, 255], [85, 255, 255], [170, 255, 255]
    ]
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    vis_parsing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) + 255
    num_of_class = np.max(parsing)
    for pi in range(1, num_of_class + 1):
        # if pi != 5 and pi != 6 and pi != 7:
        #     continue
        index = np.where(parsing == pi)
        vis_parsing[index[0], index[1], :] = part_colors[pi]
    print(vis_parsing.shape, frame.shape)
    img = cv2.addWeighted(img, 0.75, vis_parsing, 0.25, 0)
    if show_image:
        cv2.imshow('Demo', img)
        cv2.imshow('blank', vis_parsing)
    return img, parsing


# ---------------------------------------------------------------
#                     Detection Utils
# ---------------------------------------------------------------

def get_iou(base_box, compare_box):
    x1_insection = np.maximum(base_box[0] - base_box[2] / 2,
                              compare_box[:, 0] - compare_box[:, 2] / 2)
    y1_insection = np.maximum(base_box[1] - base_box[3] / 2,
                              compare_box[:, 1] - compare_box[:, 3] / 2)
    x2_insection = np.minimum(base_box[0] + base_box[2] / 2,
                              compare_box[:, 0] + compare_box[:, 2] / 2)
    y2_insection = np.minimum(base_box[1] + base_box[3] / 2,
                              compare_box[:, 1] + compare_box[:, 3] / 2)
    width_insection = np.maximum(0, x2_insection - x1_insection)
    height_insection = np.maximum(0, y2_insection - y1_insection)
    area_insection = width_insection * height_insection
    area_union = base_box[2] * base_box[3] + compare_box[:, 2] * compare_box[:, 3] - area_insection
    iou = area_insection / area_union
    return iou


def cal_result_box(overlapping):
    '''
    Take an average of the coordinates from the overlapping
    detections, weighted by their confidence scores.
    '''
    base_box = overlapping[0]
    if len(overlapping) == 1:
        return base_box
    x = 0
    y = 0
    width = 0
    height = 0
    total_score = 0
    for box in overlapping:
        score = box[4]
        x += box[0] * score
        y += box[1] * score
        width += box[2] * score
        height += box[3] * score
        total_score += score
    x = x / total_score
    y = y / total_score
    width = width / total_score
    height = height / total_score
    score = base_box[4]
    index = base_box[5]
    return [x, y, width, height, score, index]


def crop_detected_frame(img, box, landmarks, ratio=2.6, shift=0.5, use_dia=False, rotate=(False, False)):
    '''Expands and shifts the rectangle that contains the palm so that it's likely
       to cover the entire hand.
    node {
    calculator: "RectTransformationCalculator"
    input_stream: "NORM_RECT:palm_rect"
    input_stream: "IMAGE_SIZE:image_size"
    output_stream: "hand_rect_from_palm_detections"
    node_options: {
        [type.googleapis.com/mediapipe.RectTransformationCalculatorOptions] {
        scale_x: 2.6
        scale_y: 2.6
        shift_y: -0.5
        square_long: true
            }
        }
    }
    '''
    # reshape landmarks
    landmarks = landmarks.reshape(len(landmarks)//2, 2)
    if landmarks.shape == (7, 2):
        print('hand mode')
        wrist = landmarks[0]
        middle = landmarks[2]
    elif landmarks.shape == (6, 2):
        print('face mode')
        wrist = landmarks[3]
        middle = landmarks[2]
    aza = azimuth_angle(middle[0], middle[1], wrist[0], wrist[1])
    # flag: 0, [315, 360) or [0, 45),
    if 315 <= aza < 360 or 0 <= aza < 45:
        flag = 0
    # flag: -1, [45, 135), rotate counterclockwise 90 once
    elif 45 <= aza < 135:
        flag = -1
    # flag: 2, [135, 225], rotate clockwise 90 twice
    elif 135 <= aza < 225:
        flag = 2
    # flag: 1, [225, 315), rotate clockwise 90 once
    elif 225 <= aza < 315:
        flag = 1

    # init
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    # !DEBUG
    if 0:
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
        cv2.imshow('raw handboxes', img[:, :, ::-1])
        # print(xmin, ymin, xmax, ymax)
        print('raw wid:{:.3f} height:{:.3f}'.format(xmax-xmin, ymax-ymin))

    width = xmax - xmin
    height = ymax - ymin

    # expand ratio
    if use_dia:
        # 斜边
        new_len = np.sqrt(height * height + width * width) * ratio
    else:
        # 最长边长
        new_len = max(height, width) * ratio
    xmin = xmin + width * 0.5 - new_len * 0.5
    ymin = ymin + height * 0.5 - new_len * 0.5
    xmax = xmin + new_len
    ymax = ymin + new_len

    # # shift by flag
    if flag == 0:
        ymin = ymin - height*shift
        ymax = ymax - height*shift
    elif flag == -1:
        xmin = xmin - width*shift
        xmax = xmax - width*shift
    elif flag == 2:
        ymin = ymin + height*shift
        ymax = ymax + height*shift
    elif flag == 1:
        xmin = xmin + width*shift
        xmax = xmax + width*shift

    # edge cases
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img.shape[1], xmax)
    ymax = min(img.shape[0], ymax)

    crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    if rotate[0]:
        # * roughly pose align
        if not rotate[1]:
            crop_img = np.rot90(crop_img, k=flag)
        # * precisely pose align
        else:
            rotate_align = -aza
            trans = get_affine_transform(center=np.array([(xmin+xmax)/2, (ymin+ymax)/2]),
                                         scale=np.array([new_len, new_len]),
                                         rot=rotate_align,
                                         output_size=np.array([new_len, new_len]),
                                         )
            crop_img = cv2.warpAffine(
                img,
                trans, (int(new_len), int(new_len)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT)
            flag = aza

    # !DEBUG
    if 1:
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
        # cv2.imshow('expanded handboxes', img)
        cv2.imshow('crop', crop_img)
        # print(xmin, ymin, xmax, ymax)
        print('postprocess wid:{:.3f} height:{:.3f}'.format(xmax-xmin, ymax-ymin))
    return crop_img, flag, (xmin, ymin, xmax, ymax)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """

    def _get_3rd_point(a, b):
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): point(x,y)
            b (np.ndarray): point(x,y)

        Returns:
            np.ndarray: The 3rd point.
        """
        assert len(a) == 2
        assert len(b) == 2
        direction = a - b
        third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

        return third_pt

    def rotate_point(pt, angle_rad):
        """Rotate a point by an angle.

        Args:
            pt (list[float]): 2 dimensional point to be rotated
            angle_rad (float): rotation angle by radian

        Returns:
            list[float]: Rotated point.
        """
        assert len(pt) == 2
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        new_x = pt[0] * cs - pt[1] * sn
        new_y = pt[0] * sn + pt[1] * cs
        rotated_pt = [new_x, new_y]

        return rotated_pt

    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    # print(f"{src_w=}, {dst_w=}, {dst_h=}")
    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

# ---------------------------------------------------------------
#                     Keypoint Utils
# ---------------------------------------------------------------


def recover_landmark(outs, h_scale, w_scale, remove_padding=True, rotated=(False, False)):
    '''
    recover size, rotation, position
    '''
    for out, frame, flag, expanded_box in outs:
        landmark = out[0]

        if rotated[0]:
            if not rotated[1]:
                # * ROTATION rough align
                landmark = keypoint_rot(landmark, flag)
            else:
                # * ROTATION precise align
                theta = np.deg2rad(-flag)
                x_c, y_c = w_scale/2, h_scale/2
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                trans_mat_1 = np.array([[1, 0, x_c],
                                        [0, 1, y_c],
                                        [0, 0, 1], ])
                rot_mat = np.array([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta,  0],
                                    [0,         0,          1]])
                trans_mat_2 = np.array([[1, 0, -x_c],
                                        [0, 1, -y_c],
                                        [0, 0, 1], ])
                temp_kpts = np.ones_like(landmark)
                temp_kpts[:, :2] = landmark[:, :2]
                landmark[:, :2] = (trans_mat_1 @ rot_mat @ trans_mat_2 @ temp_kpts.T).T[:, :2]

        # * SIZE DONE
        height = expanded_box[3]-expanded_box[1]
        width = expanded_box[2]-expanded_box[0]
        pad_size = [width, height]
        if remove_padding:
            pad_size = [max(width, height)] * 2
        landmark[:, 0] = landmark[:, 0] / w_scale * pad_size[0] - (pad_size[0] - width) / 2.
        landmark[:, 1] = landmark[:, 1] / h_scale * pad_size[1] - (pad_size[1] - height) / 2.

        # * POSITION DONE
        landmark[:, 0] += expanded_box[0]
        landmark[:, 1] += expanded_box[1]


def draw_hand_landmarks(ori_img, outs, show_image=False):
    '''
    [0: wrist, (1,2,3,4): thumb,
    (5,6,7,8): index_finger, (9,10,11,12): middle finger,
    (13,14,15,16): ring_finger, (17,18,19,20): pinky
    (0,5,9,13,17): palm
    '''
    img = cv2.cvtColor(ori_img.copy(), cv2.COLOR_RGB2BGR)
    confidence, handedness = None, None
    landmarks = []
    for out, _, _, bbox in outs:
        try:
            landmark, confidence, handedness = out
            if confidence < 0.90:
                return
        except:
            landmark, = out

        deep = None
        if confidence and handedness:
            # deep
            deep = min_max_normalize(landmark[:, 2])
        # link landmark
        # thumb
        img = link_points(img, (0, 1, 2, 3, 4), landmark, deep)
        # index
        img = link_points(img, (5, 6, 7, 8), landmark, deep)
        # middle
        img = link_points(img, (9, 10, 11, 12), landmark, deep)
        # ring
        img = link_points(img, (13, 14, 15, 16), landmark, deep)
        # pinky
        img = link_points(img, (17, 18, 19, 20), landmark, deep)
        # palm
        img = link_points(img, (0, 5, 9, 13, 17, 0), landmark, deep)

        # Gestures
        # gesture = simpleGesture(landmark)

        # plot bbox
        xmin, ymin, xmax, ymax = bbox
        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (225, 225, 225), 4)

        # plot landmark
        for i in range(21):
            if confidence and handedness:
                p_color = 255*deep[i], 255*deep[i], 255*deep[i]
                p_radius = int(15 * (1-deep[i]))
            else:
                p_color = (0, 0, 255)
                p_radius = 8
            # img = cv2.circle(img, (int(landmark[i, 0]), int(landmark[i, 1])), 1, p_color, -1)
            img = cv2.circle(img, (int(landmark[i, 0]), int(landmark[i, 1])), p_radius, p_color, -1)
        if confidence and handedness:
            hand = 'right' if handedness > 0.5 else 'left'
            img = cv2.putText(img, str(hand),
                              (int(landmark[0, 0]), int(landmark[0, 1])), 0, 1, (255, 0, 0), 2)
            img = cv2.putText(img, '{:.3f}'.format(confidence),
                              (int(landmark[0, 0]), int(landmark[0, 1])+50), 0, 1, (255, 0, 0), 2)
        landmarks.append(landmark)

    if show_image:
        if confidence and handedness:
            if 0:
                img_3d = plot_3d_landmarks(outs)
                img = np.hstack([img, cv2.resize(img_3d, (img.shape[0], img.shape[0]))])
        cv2.imshow('Demo', img)
    return landmarks, img


def link_points(img, keypoints_set, landmark, deep=None):
    for i in range(1, len(keypoints_set)):
        a = keypoints_set[i-1]
        b = keypoints_set[i]
        if isinstance(deep, np.ndarray):
            line_color = (255*np.mean([deep[a], deep[b]]), 255 *
                          np.mean([deep[a], deep[b]]), 255*np.mean([deep[a], deep[b]]))
        else:
            line_color = (0, 255, 0)
        img = cv2.line(img, (int(landmark[a, 0]), int(landmark[a, 1])),
                       (int(landmark[b, 0]), int(landmark[b, 1])), line_color, 4)
    return img


def plot_3d_landmarks(outs):
    '''
    [0: wrist, (1,2,3,4): thumb,
    (5,6,7,8): index_finger, (9,10,11,12): middle finger,
    (13,14,15,16): ring_finger, (17,18,19,20): pinky
    (0,5,9,13,17): palm
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7, 7), dpi=200)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = Axes3D(fig)
    ax.view_init(10, 60)

    for out, _, _, _ in outs:
        landmark, _, _ = out

        temp_landmark = landmark.copy()

        x = 1280-temp_landmark[:, 0]
        y = 720-temp_landmark[:, 1]
        z = temp_landmark[:, 2]

        ax.scatter3D(x, z, y, c=z)

        ax.set_xlim(0, 1280)
        ax.set_ylim(-100, 300)
        ax.set_zlim(0, 800)
        # link-points
        lined_set = [0, 1, 2, 3, 4]
        plot_3d_line(ax, lined_set, x, z, y, (255/255., 128/255., 0/255.))
        lined_set = [0, 5, 6, 7, 8]
        plot_3d_line(ax, lined_set, x, z, y, (255/255, 153/255, 255/255))
        lined_set = [0, 9, 10, 11, 12]
        plot_3d_line(ax, lined_set, x, z, y, (102/255, 178/255, 255/255))
        lined_set = [0, 13, 14, 15, 16]
        plot_3d_line(ax, lined_set, x, z, y, (255/255, 51/255, 51/255))
        lined_set = [0, 17, 18, 19, 20]
        plot_3d_line(ax, lined_set, x, z, y, (0/255, 255/255, 0/255))

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        fig.canvas.draw()
    # plt.pause(0.01)
    vis_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    vis_img = vis_img.reshape((2800, 2800, 3))
    plt.clf()
    return vis_img.copy()


def plot_3d_line(ax, lined_set, x, y, z, color):
    for i in range(1, len(lined_set)):
        a = lined_set[i-1]
        b = lined_set[i]
        ax.plot(xs=[x[a], x[b]], ys=[y[a], y[b]], zs=[z[a], z[b]], color=color)


# ---------------------------------------------------------------
#                     Regression Utils
# ---------------------------------------------------------------


def draw_regression(frame, outs, show_image=False):
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    for out, bbox in outs:
        xmin, ymin, xmax, ymax = bbox
        img = cv2.rectangle(img,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (255, 255, 255), 2, 2)
        img = cv2.putText(img, str(out),
                          (int(xmin), int(ymin)-25), 0, 1, (255, 255, 255), 1)
    if show_image:
        cv2.imshow('Demo', img)
    return img


# ---------------------------------------------------------------
#                     Others Utils
# ---------------------------------------------------------------

def draw_classification(frame, outs, show_image=False):
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    for out, bbox in outs:
        xmin, ymin, xmax, ymax = bbox
        _, score, label = out
        img = cv2.rectangle(img,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (255, 255, 255), 2, 2)
        img = cv2.putText(img, "{}: {}".format(label, str(score)),
                          (int(xmin), int(ymin)-25), 0, 1, (255, 255, 255), 1)
    if show_image:
        cv2.imshow('Demo', img)
    return img


def draw_multimodel_func(ori_img, outs, flag, rotated, show_image=False, h_scale=224, w_scale=224):
    if flag == 'parsing':
        # recover from multiple outs
        parsing = recover_mask(ori_img, outs, remove_padding=True)
        result = draw_mask(ori_img, parsing)
    elif flag == 'landmark':
        # recover from multiple outs
        print(h_scale, w_scale)
        recover_landmark(outs, h_scale, w_scale, remove_padding=True, rotated=rotated)
        result = draw_hand_landmarks(ori_img, outs, show_image)
    elif flag == 'cls':
        result = draw_classification(ori_img, outs, show_image)
    elif flag == 'regression':
        result = draw_regression(ori_img, outs, show_image)
    else:
        result = None
        if show_image:
            cv2.imshow('Demo', ori_img[:, :, ::-1])
    return result


def random_name():
    return ''.join(random.sample(string.ascii_letters + string.digits, 5))


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def keypoint_rot_prev(keypoints, flag, frame):
    # get theta
    angle_dic = {0: np.pi/180.0*0,
                 1: np.pi/180.0*90,
                 2: np.pi/180.0*180,
                 -1: np.pi/180.0*-90}
    theta = angle_dic[flag]
    # get center and scaler
    scaler = np.stack([frame.shape[0], frame.shape[1]], axis=0)
    center = np.reshape(scaler*0.5, [1, 2])
    # rotation matrix
    rotation = np.stack([np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)], axis=0)
    rotation_matrix = np.reshape(rotation, [2, 2])
    keypoints[:, :2] = np.matmul(keypoints[:, :2]-center, rotation_matrix) + center
    return keypoints


def keypoint_rot(keypoints, flag):
    """    
    float anchr = Palm_Landmark_Model_Input_WH/2.0;
    float rotate = -palm.pose*(M_PI_2);
    for (short i = 0; i < Palm_Landmark_Model_Output_Count; ++i) {
        //输出的xy是在224上的坐标
        x = result[i*Palm_Landmark_Model_Output_Step+0];
        y = result[i*Palm_Landmark_Model_Output_Step+1];
        z = result[i*Palm_Landmark_Model_Output_Step+2];
        //cv::circle(test_img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255, 255), -1);
        if (palm.pose > 0) {
            x1 = x - anchr;
            y1 = y - anchr;
            x = x1*cos(rotate) - y1*sin(rotate) + anchr;
            y = y1*cos(rotate) + x1*sin(rotate) + anchr;
        }
        x = x / Palm_Landmark_Model_Input_WH * pad_size - ((pad_size - w) / 2.0) + cropBox[0];
        y = y / Palm_Landmark_Model_Input_WH * pad_size - ((pad_size - h) / 2.0) + cropBox[1];
    """
    angle_dic = {
        0: 0*np.pi/2.0,      # 0
        1: -3*np.pi/2.0,      # 90
        2: -2*np.pi/2.0,      # 180
        -1: -1*np.pi/2.0      # 270
    }
    anchor = 112.0
    rotate = angle_dic[flag]
    if flag != 0:
        temp_x = keypoints[:, 0] - anchor
        temp_y = keypoints[:, 1] - anchor
        keypoints[:, 0] = temp_x*np.cos(rotate) - temp_y*np.sin(rotate) + anchor
        keypoints[:, 1] = temp_y*np.cos(rotate) + temp_x*np.sin(rotate) + anchor
    return keypoints


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def recover_zero_padding(frame, post_out, ratio=(1, 1), rotate=True):
    '''
    mask remove zero padding
    '''
    h, w, _ = frame.shape

    def recover_single(out):
        # rotated
        if w > h and rotate:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # rotated and height:width > 16:9
            h_o, w_o = out.shape[:2]
            if w / h > ratio[0]/ratio[1]:
                ll = int(h_o - h * w_o/w)
                return out[ll//2:-(ll-ll//2), :]
            # rotated and height:width <= 16:9
            elif w / h < ratio[0]/ratio[1]:
                ll = int(w_o - w * h_o/h)
                return out[:, ll//2:-(ll-ll//2)]
            else:
                return out
        else:
            h_o, w_o = out.shape[:2]
            # height:width > 16:9
            if h / w > ratio[0]/ratio[1]:
                ll = int(w_o - w * h_o/h)
                return out[:, ll//2:-(ll-ll//2)]
            # height:width <= 16:9
            elif h / w < ratio[0]/ratio[1]:
                ll = int(h_o - h * w_o/w)
                return out[ll//2:-(ll-ll//2), :]
            else:
                return out
    if isinstance(post_out, (list, tuple)):
        result = []
        for out in post_out:
            result.append(recover_single(out))
    else:
        result = recover_single(post_out)
    return result


def zero_resize_padding(img, h_scale, w_scale, ratio=(1, 1), border_mode=cv2.BORDER_CONSTANT, rotate=False):
    """Zero Resize Padding

    Args:
        img (np.ndarray): input image
        h_scale (int): image resize height for model
        w_scale (int): image resize width for model
        ratio (tuple, optional): ratio of input image in (height, width). Defaults to (1, 1).
        border_mode ([type], optional): mode for zero padding. Defaults to cv2.BORDER_CONSTANT.
        rotate (bool, optional): using dynamic portrait rotate for portrait mode. Defaults to False.

    Returns:
        image: output zero padding resized image
    """
    # zero padding
    h, w, _ = img.shape
    if w > h:
        ratio = ratio[::-1]
    tb = int(ratio[0]/ratio[1]*w) - h
    lr = int(ratio[1]/ratio[0]*h) - w
    if tb >= lr:
        tb //= 2
        image = cv2.copyMakeBorder(img, abs(tb), abs(tb), 0, 0, border_mode)
    else:
        lr //= 2
        image = cv2.copyMakeBorder(img, 0, 0, abs(lr), abs(lr), border_mode)

    if w > h and rotate:
        image = cv2.resize(image, (int(h_scale), int(w_scale)))
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        image = cv2.resize(image, (int(w_scale), int(h_scale)))

    return image


def azimuth_angle(x1, y1, x2, y2):
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    # if x2 == x1:
    #     if y2 == y1:
    #         angle = 0.0
    #     elif y2 < y1:
    #         angle = 3 * math.pi / 2.0
    if x2 == x1:
        if y2 >= y1:
            angle = 0.0
        else:
            angle = math.pi
    elif y1 == y2:
        if x1 < x2:
            angle = math.pi / 2.0
        elif x1 > x2:
            angle = 3 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


def __getEuclideanDistance(posA, posB):
    return math.sqrt((posA[0] - posB[0])**2 + (posA[1] - posB[1])**2)


def __isThumbNearIndexFinger(thumbPos, indexPos):
    return __getEuclideanDistance(thumbPos, indexPos) < 0.1


def simpleGesture(landmark):
    '''
    wrist: 0
    thumb: 1,2,3,4
    index: 5,6,7,8
    middle: 9,10,11,12 
    ring: 13,14,15,16
    pinky: 17,18,19,20
    '''

    thumbIsOpen = False
    indexIsOpen = False
    middelIsOpen = False
    ringIsOpen = False
    pinkyIsOpen = False

    pseudoFixKeyPoint = landmark[1][0]
    if (landmark[3][0] < landmark[4][0] and landmark[2][0] < landmark[3][0] and pseudoFixKeyPoint < landmark[2][0] and landmark[0][0] < pseudoFixKeyPoint) or (landmark[3][0] > landmark[4][0] and landmark[2][0] > landmark[3][0] and pseudoFixKeyPoint > landmark[2][0] and landmark[0][0] > pseudoFixKeyPoint):
        thumbIsOpen = True

    pseudoFixKeyPoint = landmark[6][1]
    if landmark[7][1] < pseudoFixKeyPoint and landmark[8][1] < pseudoFixKeyPoint:
        indexIsOpen = True

    pseudoFixKeyPoint = landmark[10][1]
    if landmark[11][1] < pseudoFixKeyPoint and landmark[12][1] < pseudoFixKeyPoint:
        middelIsOpen = True

    pseudoFixKeyPoint = landmark[14][1]
    if landmark[15][1] < pseudoFixKeyPoint and landmark[16][1] < pseudoFixKeyPoint:
        ringIsOpen = True

    pseudoFixKeyPoint = landmark[18][1]
    if landmark[19][1] < pseudoFixKeyPoint and landmark[20][1] < pseudoFixKeyPoint:
        pinkyIsOpen = True

    if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        print("FIVE!")
        return "FIVE!"

    elif not middelIsOpen and not ringIsOpen and not pinkyIsOpen and __isThumbNearIndexFinger(landmark[3], landmark[7]):
        print("HEART!")
        return "HEART!"

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

    elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        print("SPIDERMAN!")
        return "SPIDERMAN!"

    elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        print("FIST!")
        return "FIST!"

    elif not indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen and __isThumbNearIndexFinger(landmark[4], landmark[8]):
        print("OK!")
        return "OK!"

    print("FingerState: thumbIsOpen? " + str(thumbIsOpen) + " - indexIsOpen? " + str(indexIsOpen) + " - middelIsOpen? " +
          str(middelIsOpen) + " - ringIsOpen? " + str(ringIsOpen) + " - pinkyIsOpen? " + str(pinkyIsOpen))


# ----------------------------------------------------------------
# optical flow for segmentation
# ----------------------------------------------------------------

# coding: utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """计算光流跟踪匹配点和光流图
    输入参数:
        pre_gray: 上一帧灰度图
        cur_gray: 当前帧灰度图
        prev_cfd: 上一帧光流图
        dl_weights: 融合权重图
        disflow: 光流数据结构
    返回值:
        is_track: 光流点跟踪二值图，即是否具有光流点匹配
        track_cfd: 光流跟踪图
    """
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    flow_fw = np.round(flow_fw).astype(np.int)
    flow_bw = np.round(flow_bw).astype(np.int)
    y_list = np.array(range(h))
    x_list = np.array(range(w))
    yv, xv = np.meshgrid(y_list, x_list)
    yv, xv = yv.T, xv.T
    cur_x = xv + flow_fw[:, :, 0]
    cur_y = yv + flow_fw[:, :, 1]

    # 超出边界不跟踪
    not_track = (cur_x < 0) + (cur_x >= w) + (cur_y < 0) + (cur_y >= h)
    flow_bw[~not_track] = flow_bw[cur_y[~not_track], cur_x[~not_track]]
    not_track += (np.square(flow_fw[:, :, 0] + flow_bw[:, :, 0]) +
                  np.square(flow_fw[:, :, 1] + flow_bw[:, :, 1])) >= check_thres
    track_cfd[cur_y[~not_track], cur_x[~not_track]] = prev_cfd[~not_track]

    is_track[cur_y[~not_track], cur_x[~not_track]] = 1

    not_flow = np.all(
        np.abs(flow_fw) == 0, axis=-1) * np.all(
            np.abs(flow_bw) == 0, axis=-1)
    dl_weights[cur_y[not_flow], cur_x[not_flow]] = 0.05
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    fusion_cfd = dl_cfd.copy()
    is_track = is_track.astype(np.bool)
    fusion_cfd[is_track] = dl_weights[is_track] * dl_cfd[is_track] + (
        1 - dl_weights[is_track]) * track_cfd[is_track]
    # 确定区域
    index_certain = ((dl_cfd > 0.9) + (dl_cfd < 0.1)) * is_track
    index_less01 = (dl_weights < 0.1) * index_certain
    fusion_cfd[index_less01] = 0.85 * dl_cfd[index_less01] + 0.15 * track_cfd[
        index_less01]
    index_larger09 = (dl_weights >= 0.1) * index_certain
    fusion_cfd[index_larger09] = 0.85 * dl_cfd[index_larger09] + 0.15 * track_cfd[
        index_larger09]
    return fusion_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optic_flow_process(cur_gray, scoremap, prev_gray, pre_cfd, disflow,
                       is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        pre_gray : 前一帧灰度图
        pre_cfd  ：前一帧融合结果
        scoremap : 当前帧分割结果
        difflow  : 光流
        is_init : 是否第一帧
    Returns:
        fusion_cfd : 光流追踪图和预测结果融合图
    """
    h, w = scoremap.shape
    cur_cfd = scoremap.copy()

    if is_init:
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((h, w), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, pre_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)

    return fusion_cfd
