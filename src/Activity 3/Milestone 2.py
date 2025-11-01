import cv2
import numpy as np

trainCols = 50
train = x[:, :trainCols]
valid = x[:, trainCols:100]
# Next, compute the number of training/testing examples per digit
trainPerDigit = 500 / (100 / trainCols)
validPerDigit = 500 / (100 / (100 - trainCols))

train = train.reshape(-1, 400).astype(np.float32)
valid = valid.reshape(-1, 400).astype(np.float32)
print(train.shape, valid.shape)

digitCats = np.arange(10)
train_labels = np.repeat(digitCats,trainPerDigit)[:, np.newaxis]
valid = np.repeat(digitCats, validPerDigit)[:, np.newaxis]