import os
from Laser_Detector import Detector
import argparse
import json
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Laser spot detector test')
    parser.add_argument('--config', help='config file path', default='Config')
    parser.add_argument('-i', '--input', help='input path of images')
    parser.add_argument('-o', '--output', help='output path of results', default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.config) and os.path.exists(args.input)
    if not os.path.exists(os.path.dirname(args.output)):
        os.mkdir(os.path.dirname(args.output))
    with open(args.config, "r") as f:
        config = json.load(f)
    detector = Detector(**config)
    if os.path.isdir(args.input):
        for img_name in os.listdir(args.input):
            img = cv2.imread(os.path.join(args.input, img_name))
            result, _ = detector.detect(img)
            cv2.imwrite(os.path.join(os.path.dirname(args.output), 'result_' + img_name), result)
    else:
        img_name = args.input.split('/')[-1]
        img = cv2.imread(args.input)
        result, _ = detector.detect(img)
        if '.' in args.output:
            cv2.imwrite(args.output, result)
        else:
            cv2.imwrite(os.path.join(args.output, img_name), result)
