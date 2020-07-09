import cv2.cv2 as cv
import numpy as np
from utils import *

training_sets, testing_sets = load_datasets()

train_x = np.load(os.path.join("./model_cache/train_data", training_sets[0]))
test_x = np.load(os.path.join('./model_cache/test_data', testing_sets[0]))

gray_images = []
for image in train_x:
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = np.expand_dims(gray_img, axis=2)
    gray_images.append(gray_img)

np.save('./model_cache/train_data/training_sets-grayscale-full_augmentation_train_x.npy', gray_images)

gray_images = []
for image in test_x:
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_img = np.expand_dims(gray_img, axis=2)
    gray_images.append(gray_img)

np.save('./model_cache/test_data/test_set-grayscale-test_x.npy', gray_images)

