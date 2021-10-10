import numpy as np
import cv2
import os
import pandas as pd
import shutil
import joblib


def main():
    # 加载图片 读取彩色图像
    image = cv2.imread('./test_02/rgb/244.png', cv2.IMREAD_COLOR)
    # print(image)
    # cv2.imshow("image", image)
    # 图像归一化，且转换为浮点型
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    gamma = 1
    gamma_max = 10
    s = 100
    MAX_VALUE = 100
    # 调节饱和度和亮度的窗口
    cv2.namedWindow("gamma and s", cv2.WINDOW_AUTOSIZE)

    def nothing(*arg):
        pass

    # 滑动块
    cv2.createTrackbar("gamma", "gamma and s", gamma, gamma_max, nothing)
    cv2.createTrackbar("s", "gamma and s", s, MAX_VALUE, nothing)
    # 调整饱和度和亮度后的效果
    lsImg = np.zeros(image.shape, np.float32)
    # 调整饱和度和亮度
    while True:
        # 复制
        hlsCopy = np.copy(hlsImg)
        # 得到 l 和 s 的值
        gamma = cv2.getTrackbarPos('gamma', 'gamma and s')
        s = cv2.getTrackbarPos('s', 'gamma and s')
        # 1.调整亮度（线性变换) , 2.将hlsCopy[:, :, 1]和hlsCopy[:, :, 2]中大于1的全部截取
        hlsCopy[:, :, 1] = np.power(hlsCopy[:, :, 1], gamma)
        # 饱和度
        hlsCopy[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        # 显示调整后的效果
        cv2.imshow("gamma and s", lsImg)
        ch = cv2.waitKey(5)
        # 按 ESC 键退出
        if ch == 27:
            break
        elif ch == ord('s'):
            # 按 s 键保存并退出
            # 保存结果
            lsImg = lsImg * 255
            lsImg = lsImg.astype(np.uint8)
            cv2.imwrite("lsImg.jpg", lsImg)
            break
    # 关闭所有的窗口
    cv2.destroyAllWindows()


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


def Sobel_detection(img, threshold=-1):
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # 获取水平方向的偏导
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)  # 获取竖直方向的偏导
    angle = np.arctan(dx / (dy + 1e-5))
    temp = np.where(dx < 0)
    angle[temp] += np.pi
    mag = cv2.magnitude(dx, dy)
    mag = cv2.convertScaleAbs(mag)
    # if threshold == -1:
    #     _, mag = cv2.threshold(mag, 0, 255, cv2.THRESH_OTSU)
    # else:
    #     mag[mag >= threshold] = 255
    #     mag[mag < threshold] = 0
    return mag


# def preprocess():
#     grad_root = './test_mp4_grad/'
#     hsv_root = './test_mp4_hsv_7/'
#     out_root = './test_mp4_new/'
#     # img_root = './test_mp4/'
#     for img_name in os.listdir(grad_root):
#         grad_path = os.path.join(grad_root, img_name)
#         hsv_path = os.path.join(hsv_root, img_name)
#         grad = cv2.imread(grad_path, flags=cv2.IMREAD_UNCHANGED)
#         hsv = cv2.imread(hsv_path, flags=cv2.IMREAD_UNCHANGED)
#         hsv = cv2.threshold(hsv, 100, 255, cv2.THRESH_BINARY)[1]
#         img = grad.copy()
#         img[hsv==0] = 0
#         # out = gray(img_path)
#         cv2.imwrite(os.path.join(out_root, img_name), img)
def preprocess(raw_img, grad):
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([150, 20, 100])
    upper_red1 = np.array([180, 255, 255])
    thresh = cv2.inRange(img, lower_red1, upper_red1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)[1]
    img = grad.copy()
    img[thresh == 0] = 0
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return img


def Variance_compute(center, contours):
    points = contours.reshape((-1, 2))
    distance = np.linalg.norm(points - center, axis=1)
    variance = np.var(distance)
    return variance


def Circle_detect(img, minVar, minRadius, maxRadius, is_circle=True):
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


def candidate(frame_):
    frame = frame_.copy()
    img = adjust(frame)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Sobel = Sobel_detection(gray_img)
    grad = preprocess(frame, Sobel)
    circles = Circle_detect(grad, 5, minRadius=2, maxRadius=15, is_circle=False)
    # result = frame.copy()
    rects = []
    for i in range(len(circles)):
        # 用外接矩形框出目标点
        x, y, w, h = cv2.boundingRect(circles[i])
        new_x = max(0, x - w)
        new_y = max(0, y - h)
        rects.append((new_x, new_y, 3 * w, 3 * h))
        # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # # 用圆形框出目标点
        # x, y = int(circles[i][0]), int(circles[i][1])
        # radius = int(circles[i][2])
        # result = cv2.circle(result, (x, y), radius, (0, 255, 0))

    # cv2.imwrite('test_gray.jpg', gray_img)
    # cv2.imwrite('test_Sobel.jpg', Sobel)
    # cv2.imwrite('test_grad.jpg', grad)
    # cv2.imwrite('test_out.jpg', result)
    return rects


def DataSet_build(frame, rects, name):
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        x_ = min(frame.shape[1], x + w)
        y_ = min(frame.shape[0], y + h)
        temp = frame[y: y_, x: x_].copy()
        temp = cv2.resize(temp, (32, 32))
        cv2.imwrite(f'DataSet/img/{name}_{i}.jpg', temp)


def label():
    img_names = os.listdir('DataSet/img')
    img_names.sort()
    labels_path = 'DataSet/labels.csv'
    begin = '00_00000_000.jpg'
    if os.path.exists(labels_path):
        labels = pd.read_csv(labels_path, index_col=0)
        with open('continue.txt', 'r') as f:
            begin = f.read()
    else:
        labels = pd.Series(np.ones(len(img_names)) * 0, index=img_names, name='is_laser')
    title = "Press 'Y' if the image is laser dot or press 'N' to indicate not, press 'Q' to exit"
    # cv2.namedWindow(title, 0)
    for name in img_names:
        if name < begin:
            continue
        img = cv2.imread('DataSet/img/' + name)
        cv2.namedWindow(name, 0)
        cv2.moveWindow(name, 200, 200)
        cv2.imshow(name, img)
        is_laser = cv2.waitKey()
        cv2.destroyWindow(name)
        if is_laser == ord('y'):
            labels.loc[name] = 1
        elif is_laser == ord('n'):
            labels.loc[name] = 0
        else:
            labels.to_csv('DataSet/labels.csv')
            with open('continue.txt', 'w') as f:
                f.write(name)
            break


def contrive_label():
    labels = pd.read_csv('DataSet/labels.csv', index_col=0)
    for name in os.listdir('DataSet/img/'):
        img_path = 'DataSet/img/' + name
        if labels.loc[name][0] == 1:
            shutil.copyfile(img_path, 'DataSet/img_with_label/positive/'+name)
        elif labels.loc[name][0] == 0:
            shutil.copyfile(img_path, 'DataSet/img_with_label/negative_with_uncertainty/'+name)
            shutil.copyfile(img_path, 'DataSet/img_with_label/negative_without_uncertainty/'+name)
        elif labels.loc[name][0] == 2:
            shutil.copyfile(img_path, 'DataSet/img_with_label/negative_with_uncertainty/'+name)
        else:
            print(f"Error! The file's name is {name}")


class Ensemble():
    def __init__(self, clfs=False, load_param_path='Classifier/Ensemble.model'):
        if not clfs:
            self.clfs = clfs
        else:
            self.clfs = joblib.load(load_param_path)

    def predict(self, x):
        result = 0
        for clf in self.clfs:
            result += clf.predict(x)
        return np.sign(result)


if __name__ == "__main__":
    # for i in range(1, 12):
    #     root_name = os.path.join('ImageSet', str(i))
    #     # root_name = 'V11004-220853_grad_thresh_150'
    #     if not os.path.exists(root_name):
    #         os.makedirs(root_name)
    #     videoCapture = cv2.VideoCapture(root_name + '.mp4')
    #     fps = round(videoCapture.get(cv2.CAP_PROP_FPS))
    #     success, frame = videoCapture.read()
    #     frame_id = 0
    #     while success:
    #         # img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         # lower_red1 = np.array([150, 20, 100])
    #         # upper_red1 = np.array([180, 255, 255])
    #         # thresh = cv2.inRange(img, lower_red1, upper_red1)
    #         # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    #         if frame_id % fps == 0:
    #             rects = candidate(frame)
    #             name = f'{i}_{str(frame_id).zfill(5)}'
    #             DataSet_build(frame, rects, name)
    #             # cv2.imwrite(os.path.join(root_name, str(frame_id)) + '.jpg', result)
    #         frame_id += 1
    #         success, frame = videoCapture.read()  # 获取下一帧
    #     videoCapture.release()
    contrive_label()
    # img = cv2.imread('55.png')
    # result = candidate(img)
    # cv2.namedWindow('temp', 0)
    # cv2.imshow('temp', result)
    # cv2.waitKey()

    # img = cv2.imread(os.path.join(img_root, img_name), cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_red1 = np.array([150, 20, 100])
    # upper_red1 = np.array([180, 255, 255])
    # thresh = cv2.inRange(img, lower_red1, upper_red1)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # thresh3 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # thresh5 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # thresh7 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # cv2.imwrite('HSV_thresh.jpg', thresh)
    # cv2.imwrite('HSV_thresh3.jpg', thresh3)
    # cv2.imwrite('HSV_thresh5.jpg', thresh5)
    # cv2.imwrite('HSV_thresh7.jpg', thresh7)

    # for img_name in os.listdir(img_root):
    #     img_path = os.path.join(img_root, img_name)
    #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     Sobel = Sobel_detection(img)
    #     cv2.imwrite(os.path.join(out_root, img_name),Sobel)
    # grad = cv2.imread('./test_output/2.jpg', cv2.IMREAD_GRAYSCALE)
    # grad[grad<100]=0
    # grad[grad>=100] = 255
    # cv2.namedWindow('s', 0)
    # cv2.imshow('s',grad)
    # cv2.waitKey()
