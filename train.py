import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as Fc
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib


def img_trans(img, device):
    trans = T.ToTensor()
    input_img = trans(img).unsqueeze_(0)
    return input_img.to(device)


def augmentation(img, name):
    if name[-4:] == '.jpg':
        file_name = name[:-4]
    else:
        file_name = name
    rotated = [Fc.rotate(img, angle) for angle in range(0, 360, 45)]
    for i, angle in enumerate(range(0, 360, 45)):
        rotated[i].save(f'DataSet/img_with_label/positive_augmented/{file_name}_{str(angle).zfill(3)}.jpg', quality=95)


def positive_feature():
    img_names = os.listdir('DataSet/img_with_label/positive_augmented')
    feature_path = 'DataSet/img_with_label/positive_augmented.csv'
    if not os.path.exists(feature_path):
        dataset = pd.DataFrame(np.ones((len(img_names), 1001)), index=img_names)
        dataset = dataset.rename(columns={0: 'label'})
    else:
        dataset = pd.read_csv(feature_path, index_col=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    for name in img_names:
        img = Image.open('DataSet/img_with_label/positive_augmented/' + name).convert('RGB')
        # augmentation(img, name)
        input = img_trans(img, device)
        output = model(input)
        feature = output.cpu().detach().numpy()
        dataset.loc[name, 1:1000] = feature.copy()
    dataset.to_csv(feature_path)


def negative_feature():
    img_names = os.listdir('DataSet/img_with_label/negative_with_uncertainty')
    feature_path = 'DataSet/img_with_label/negative_with_uncertainty.csv'
    if not os.path.exists(feature_path):
        dataset = pd.DataFrame(np.zeros((len(img_names), 1001)), index=img_names)
        dataset = dataset.rename(columns={0: 'label'})
    else:
        dataset = pd.read_csv(feature_path, index_col=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    for name in img_names:
        img = Image.open('DataSet/img_with_label/negative_with_uncertainty/' + name).convert('RGB')
        # augmentation(img, name)
        input = img_trans(img, device)
        output = model(input)
        feature = output.cpu().detach().numpy()
        dataset.loc[name, 1:1000] = feature.copy()
    dataset.to_csv(feature_path)


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


def feature_preprocess(under_sample=5):
    positive = pd.read_csv('DataSet/img_with_label/positive_augmented.csv', index_col=0)
    negative = pd.read_csv('DataSet/img_with_label/negative_with_uncertainty.csv', index_col=0)
    negative.loc[:, 'label'] = -1
    train_positive, test_positive = train_test_split(positive, test_size=0.2)
    train_negative, test_negative = train_test_split(negative, test_size=0.2)
    clfs = []
    for i in range(under_sample):
        clf = AdaBoostClassifier(base_estimator=LinearSVC(), n_estimators=5, algorithm='SAMME')
        train_sub_nagetive = train_negative.sample(n=train_positive.shape[0], replace=True)
        combined_data = pd.concat((train_positive, train_sub_nagetive))
        combined_data = combined_data.sample(frac=1)
        X = combined_data.iloc[:, 1:]
        y = combined_data.iloc[:, 0]
        clf.fit(X, y)
        clfs.append(clf)
    final_clf = Ensemble(clfs)
    test = pd.concat((test_positive, test_negative)).sample(frac=1)
    y_predict = final_clf.predict(test.iloc[:, 1:])
    print(f'正确率:{accuracy_score(test.iloc[:, 0], y_predict):.2f}')


if __name__ == '__main__':
    # negative_feature()
    feature_preprocess()
