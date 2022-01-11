import torchvision.transforms as T
import torchvision.transforms.functional as Fc
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib
from Laser_Detector import Detector
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Laser spot detector test')
    parser.add_argument('--config', help='Config file path', default='Config')
    parser.add_argument('-p', '--positive_path', help='Input path of positive training images')
    parser.add_argument('-n', '--negative_path', help='Input path of negative training images')
    parser.add_argument('--max_iter', type=int, help='The number of iterations')
    parser.add_argument('-e', '--extract', action='store_true', help='Whether to extract image features')
    parser.add_argument('-o', '--output', help='Output path of ensemble model')
    args = parser.parse_args()
    return args


def img_trans(img, device):
    trans = T.ToTensor()
    input_img = trans(img).unsqueeze_(0)
    return input_img.to(device)


def augmentation(img, name, output_path):
    if name[-4:] == '.jpg':
        file_name = name[:-4]
    else:
        file_name = name
    rotated = [Fc.rotate(img, angle) for angle in range(0, 360, 45)]
    for i, angle in enumerate(range(0, 360, 45)):
        rotated[i].save(os.path.join(output_path, file_name + '_' + str(angle).zfill(3) + '.jpg'), quality=95)


def extract_feature(path, detector):
    img_names = os.listdir(path)
    feature_path = f'{path}.csv'
    if not os.path.exists(feature_path):
        dataset = pd.DataFrame(np.ones((len(img_names), 1001)), index=img_names)
        dataset = dataset.rename(columns={0: 'label'})
    else:
        dataset = pd.read_csv(feature_path, index_col=0)
    for name in img_names:
        img = Image.open(os.path.join(path, name)).convert('RGB')
        input = img_trans(img, detector.device)
        output = detector.model(input)
        feature = output.cpu().detach().numpy()
        dataset.loc[name, 1:1000] = feature.copy()
    dataset.to_csv(feature_path)


def feature_preprocess(p_path, n_path, detector, output_path, max_iter, under_sample=5):
    p_features_path = p_path + '.csv'
    n_features_path = n_path + '.csv'
    positive = pd.read_csv(p_features_path, index_col=0)
    negative = pd.read_csv(n_features_path, index_col=0)
    negative.loc[:, 'label'] = -1
    train_positive, test_positive = train_test_split(positive, test_size=0.2)
    train_negative, test_negative = train_test_split(negative, test_size=0.2)
    clfs = []
    for i in range(under_sample):
        clf = AdaBoostClassifier(base_estimator=LinearSVC(max_iter=max_iter), n_estimators=5, algorithm='SAMME')
        train_sub_negative = train_negative.sample(n=train_positive.shape[0], replace=True)
        combined_data = pd.concat((train_positive, train_sub_negative))
        combined_data = combined_data.sample(frac=1)
        X = combined_data.iloc[:, 1:]
        y = combined_data.iloc[:, 0]
        clf.fit(X, y)
        clfs.append(clf)
    detector.clf.assign(clfs)
    joblib.dump(clfs, output_path)
    test = pd.concat((test_positive, test_negative)).sample(frac=1)
    y_predict = detector.clf.predict(test.iloc[:, 1:])
    print(f'Accuracy:{accuracy_score(test.iloc[:, 0], y_predict):.2f}')


if __name__ == '__main__':
    args = parse_args()
    assert os.path.exists(args.config) and os.path.exists(args.positive_path) and os.path.exists(args.negative_path)
    with open(args.config, "r") as f:
        config = json.load(f)
    detector = Detector(**config)
    if args.extract:
        extract_feature(args.positive_path, detector)
        extract_feature(args.negative_path, detector)
    feature_preprocess(args.positive_path, args.negative_path, detector, args.output, args.max_iter)
