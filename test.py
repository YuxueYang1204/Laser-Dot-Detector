import os
from Laser_Detector import Detector
import argparse
import json
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Laser spot detector test')
    parser.add_argument('--config', help='config file path', default='Config')
    parser.add_argument('-i', '--input', help='input path of test images')
    parser.add_argument('-o', '--output', help='output path of test results', default=None)
    args = parser.parse_args()
    return args


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


def check_accuracy(input_path, output_path, detector):
    if output_path is not None and not os.path.exists(output_path):
        os.mkdir(output_path)
    TP = 0
    FP = 0
    FN = 0
    for path in os.listdir(os.path.join(input_path, 'GroundTruth')):
        img_name = path.split('.')[0] + '.jpg'
        img = cv2.imread(os.path.join(input_path, 'TestData', img_name))
        result, rects = detector.detect(img)
        if output_path is not None:
            cv2.imwrite(os.path.join(output_path, img_name), result)
        with open(os.path.join(input_path, 'GroundTruth', path)) as f:
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
        print(img_name)
    print(f'TP:{TP}\nFP:{FP}\nFN:{FN}\nPrecision:{TP / (TP + FP)}\nRecall:{TP / (TP + FN)}')


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.config) and os.path.exists(args.input)
    with open(args.config, "r") as f:
        config = json.load(f)
    detector = Detector(**config)
    check_accuracy(args.input, args.output, detector)
