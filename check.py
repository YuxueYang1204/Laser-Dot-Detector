import cv2
import os
import json
import numpy as np
import joblib
import torchvision.models as models
import torchvision.transforms as T
import torch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from PIL import Image


class Ensemble():
    def __init__(self, clfs=False, load_param_path='Classifier/Ensemble.model'):
        if clfs:
            self.clfs = clfs
        else:
            self.clfs = joblib.load(load_param_path)

    def predict(self, x):
        result = 0
        for clf in self.clfs:
            result += clf.predict(x)
        return np.sign(result)


def adjust(image):
    # 加载图片 读取彩色图像
    # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print(image)
    # cv2.imshow("image", image)
    # 图像归一化，且转换为浮点型
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    gamma = 3
    s = 100
    MAX_VALUE = 100
    # 调节饱和度和亮度的窗口
    lsImg = np.zeros(image.shape, np.float32)
    hlsCopy = np.copy(hlsImg)
    # 得到 l 和 s 的值
    hlsCopy[:, :, 1] = np.power(hlsCopy[:, :, 1], gamma)
    # 饱和度
    hlsCopy[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    lsImg = lsImg * 255
    return lsImg.astype(np.uint8)


def gray(img_path):
    img = cv2.imread(img_path)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return out


def Sobel_detection(img):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 获取水平方向的偏导
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)  # 获取竖直方向的偏导
    mag = cv2.magnitude(dx, dy)
    mag = cv2.convertScaleAbs(mag)
    return mag


def preprocess(raw_img, grad):
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([120, 0, 220])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([0, 0, 230])
    upper_red2 = np.array([10, 255, 255])
    thresh = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # lower_red1 = np.array([120, 0, 220])
    # upper_red1 = np.array([180, 255, 255])
    # thresh = cv2.inRange(hsv, lower_red1, upper_red1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    img = grad.copy()
    img[thresh == 0] = 0
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    return img


def Variance_compute(center, contours):
    points = contours.reshape((-1, 2))
    distance = np.linalg.norm(points - center, axis=1)
    variance = np.var(distance)
    return variance


def Circle_detect(img, minVar, minRadius, maxRadius, is_circle):
    result = []
    # 对二值图像生成边界
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if is_circle:
        for i in range(len(contours)):
            # 根据边界情况得到最小圆的圆心和半径
            center, radius = cv2.minEnclosingCircle(contours[i])
            # 计算边界上各点至最小圆的圆心的距离方差，当方差小于minVar使可以认为符合圆形检测
            variance = Variance_compute(center, contours[i])
            if variance < minVar and minRadius < radius < maxRadius:
                result.append((*center, radius))
    else:
        for i in range(len(contours)):
            # 根据边界情况得到最小圆的圆心和半径
            center, radius = cv2.minEnclosingCircle(contours[i])
            # 计算边界上各点至最小圆的圆心的距离方差，当方差小于minVar使可以认为符合圆形检测
            variance = Variance_compute(center, contours[i])
            if variance < minVar and minRadius < radius < maxRadius:
                result.append(contours[i])
    return result


def select(frame_, device, clf, model, is_circle):
    frame = frame_.copy()
    img = adjust(frame)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Sobel = Sobel_detection(gray_img)
    grad = preprocess(frame, Sobel)
    circles = Circle_detect(grad, 6, minRadius=3, maxRadius=10, is_circle=is_circle)
    # result = np.ones_like(frame_) * 255
    result = frame.copy()
    rects = []
    for i in range(len(circles)):
        if is_circle:
            # 用圆形框出目标点
            x, y = int(circles[i][0]), int(circles[i][1])
            radius = int(circles[i][2])
            result = cv2.circle(result, (x, y), radius, (0, 255, 0), -1)
        else:
            # 用外接矩形框出目标点
            x, y, w, h = cv2.boundingRect(circles[i])
            new_x = max(0, x - w)
            new_y = max(0, y - h)
            new_w = 3 * w
            new_h = 3 * h
            new_x_ = min(frame_.shape[1], new_x + new_w)
            new_y_ = min(frame_.shape[0], new_y + new_h)
            temp = frame_[new_y: new_y_, new_x: new_x_].copy()
            temp = cv2.resize(temp, (32, 32))
            temp_img = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)).convert('RGB')
            # augmentation(img, name)
            input = img_trans(temp_img, device)
            output = model(input)
            feature = output.cpu().detach().numpy()
            if clf.predict(feature) == 1:
                rects.append((new_x, new_y, new_x_, new_y_))
                result = cv2.rectangle(result, (new_x, new_y), (new_x_, new_y_), (0, 255, 0), 2)
    return result, rects


def img_trans(img, device):
    trans = T.ToTensor()
    input_img = trans(img).unsqueeze_(0)
    return input_img.to(device)


def main(raw_img, is_circle=False):
    result, rects = select(raw_img, device, clf, model, is_circle)
    return result, rects
    # for i in range(8, 12):
    #     root_name = os.path.join('ImageSet', str(i))
    #     # root_name = 'V11004-220853_grad_thresh_150'
    #     if not os.path.exists(root_name):
    #         os.makedirs(root_name)
    #     out_path = f'output/track_{i}.mp4'
    #     video = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps=30,
    #                             frameSize=(3840, 2160))
    #     videoCapture = cv2.VideoCapture(root_name + '.mp4')
    #     fps = round(videoCapture.get(cv2.CAP_PROP_FPS))
    #     success, frame = videoCapture.read()
    #     # cv2.namedWindow('test', 0)
    #     # frame_id = 0
    #     while success:
    #         # img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         # lower_red1 = np.array([150, 20, 100])
    #         # upper_red1 = np.array([180, 255, 255])
    #         # thresh = cv2.inRange(img, lower_red1, upper_red1)
    #         # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #         # if frame_id % fps == 0:
    #         result = select(frame, device, clf)
    #         video.write(result)
    #         # cv2.imwrite(os.path.join(root_name, str(frame_id)) + '.jpg', result)
    #         # frame_id += 1
    #         success, frame = videoCapture.read()  # 获取下一帧
    #     videoCapture.release()
    #     video.release()


def laser_spot_detector():
    root_name = 'ImageSet/test'
    out_path = f'output/test_track.avi'
    video = cv2.VideoWriter(out_path, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=30,
                            frameSize=(1920, 1080))
    videoCapture = cv2.VideoCapture(root_name + '.mp4')
    # fps = round(videoCapture.get(cv2.CAP_PROP_FPS))
    success, frame = videoCapture.read()
    # cv2.namedWindow('test', 0)
    # frame_id = 0
    while success:
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower_red1 = np.array([150, 20, 100])
        # upper_red1 = np.array([180, 255, 255])
        # thresh = cv2.inRange(img, lower_red1, upper_red1)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # if frame_id % fps == 0:
        result, _ = select(frame, device, clf, model, False)
        video.write(result)
        # cv2.imwrite(os.path.join(root_name, str(frame_id)) + '.jpg', result)
        # frame_id += 1
        success, frame = videoCapture.read()  # 获取下一帧
    videoCapture.release()
    video.release()


def IOU(a1, a2, b1, b2):
    S_a = (a2[0] - a1[0]) * (a2[1] - a1[1])
    S_b = (b2[0] - b1[0]) * (b2[1] - b1[1])
    xmin = max(a1[0], b1[0])
    ymin = max(a1[1], b1[1])
    xmax = min(a2[0], b2[0])
    ymax = min(a2[1], b2[1])
    height = max(0, ymax - ymin)
    width = max(0, xmax - xmin)
    S_I = height * width
    iou = S_I / (S_a + S_b - S_I)
    return iou


def accuracy():
    TP = 0
    FP = 0
    FN = 0
    for path in os.listdir('Accurate/groundtruth'):
        # os.mkdir(f'test/{path}')
        # for img_name in os.listdir('test_img/' + path):
        # for img_name in ['0060.jpg', '0390.jpg', '0510.jpg', '0630.jpg', '0750.jpg']:
        # img = cv2.imread(f'test_img/{path}/{img_name}')
        img_name = path.split('.')[0] + '.jpg'
        img = cv2.imread(f'Accurate/ImageSet/{img_name}')
        result, rects = main(img, False)
        with open(f'Accurate/groundtruth/{path}') as f:
            groundData = json.load(f)
        for groundtruth in groundData['shapes']:
            ground_p1 = groundtruth['points'][0]
            ground_p2 = groundtruth['points'][1]
            flag = 0
            for temp in rects:
                temp_p1 = list(temp[0:2])
                temp_p2 = list(temp[2:4])
                iou = IOU(ground_p1, ground_p2, temp_p1, temp_p2)
                if iou > 0.05:
                    if flag == 0:
                        flag = 1
                        TP += 1
                else:
                    FP += 1
            if flag == 0:
                FN += 1
    print(f'TP:{TP}\nFP:{FP}\nFN:{FN}\nPrecision:{TP / (TP + FP)}\nRecall:{TP / (TP + FN)}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    clf = Ensemble()
    # for frame_id in ['0060.jpg', '0084.jpg', '0425.jpg', '0804.jpg', '0909.jpg']:
    # for frame_id in ['0909.jpg']:
    # for frame_id in ['49.png', '71.png', '110.png']:
    for frame_id in os.listdir('H:/rgb'):
        frame = cv2.imread(f'H:/rgb/{frame_id}')
        frame, _ = main(frame, False)
        cv2.imwrite(f'H:/rgb_out_6/{frame_id}', frame)
    # video = cv2.VideoCapture(f'ImageSet/test3.mp4')
    # success, frame = video.read()
    # frame_id = 0
    # while success:
    #     if frame_id in [60, 111,425, 804,924]:
    #         result, _ = main(frame, False)
    #         cv2.imwrite(f'test_img/test3_{str(frame_id).zfill(4)}.jpg', result)
    #     frame_id += 1
    #     success, frame = video.read()
    # video.release()
    # for i in range(2, 12):
    #     os.makedirs(f'test_img/{i}')
    #     video = cv2.VideoCapture(f'ImageSet/{i}.mp4')
    #     success, frame = video.read()
    #     frame_id = 0
    #     while success:
    #         if frame_id % 30 == 0:
    #             cv2.imwrite(f'test_img/{i}/{str(frame_id).zfill(4)}.jpg', frame)
    #         frame_id += 1
    #         success, frame = video.read()
    #     video.release()
    # laser_spot_detector()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = models.resnet50(pretrained=True).to(device)
    # model.eval()
    # clf = Ensemble()
    # img = cv2.imread('test_img/6/0720.jpg')
    # result, rects = main(img)
    # cv2.imwrite('CCDC_en/imgs/candidate/result_5_0330.jpg', result)
    # for path in os.listdir('test_img/8'):
