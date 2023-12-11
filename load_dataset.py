# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader for NNGP experiments.

Loading MNIST dataset with train/valid/test split as numpy array.

Usage:
mnist_data = load_dataset.load_mnist(num_train=50000, use_float64=True,
                                     mean_subtraction=True)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.utils import to_categorical

#-----------------------------------------------------------------------------------------------수정
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------------------------수정

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'C:\\Users\\user\\iCloudDrive\\git\\nngp',
                    'Directory for data.')

def load_mnist(num_train=50000,
               use_float64=False,
               mean_subtraction=False,
               random_roated_labels=False):
  """Loads MNIST as numpy array."""

  data_dir = FLAGS.data_dir
  datasets = input_data.read_data_sets(
    data_dir, False, validation_size=10000, one_hot=True)
  
  mnist_data = _select_mnist_subset(
      datasets,
      num_train,
      use_float64=use_float64,
      mean_subtraction=mean_subtraction,
      random_roated_labels=random_roated_labels)

  return mnist_data

#-----------------------------------------------------------------------------------------------수정
def load_cifar10(num_train=50000,
                 use_float64=False,
                 mean_subtraction=False,
                 random_roated_labels=False):
    """Loads CIFAR-10 as numpy array."""

    # CIFAR-10 데이터셋 다운로드 및 로드
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    

    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # 데이터 타입 설정
    data_precision = np.float64 if use_float64 else np.float32
    
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    # Convert the dataset to float32 format
    train_images = train_images.astype(data_precision)
    test_images = test_images.astype(data_precision)
    
    # scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
    # train_labels_one_hot = tf.squeeze(tf.one_hot(train_labels, 10),axis=1)
    # test_labels_one_hot = tf.squeeze(tf.one_hot(test_labels, 10),axis=1)
    # train_labels = tf.squeeze(tf.one_hot(train_labels, 10), axis=1).numpy().astype(data_precision)
    # test_labels = tf.squeeze(tf.one_hot(test_labels, 10), axis=1).numpy().astype(data_precision)
    # with tf.Session() as sess:
    #     # One-hot encoding for labels
    #     train_labels = sess.run(tf.squeeze(tf.one_hot(train_labels, 10), axis=1)).astype(data_precision)
    #     test_labels = sess.run(tf.squeeze(tf.one_hot(test_labels, 10), axis=1)).astype(data_precision)

    num_val = num_train//4
    num_train = num_train - num_val

    # 훈련 데이터셋을 훈련 및 검증 데이터셋으로 분리
    valid_images = train_images[num_train:]
    valid_labels = train_labels[num_train:]
    train_images = train_images[:num_train]
    train_labels = train_labels[:num_train]

    # 평균 빼기 전처리
    # if mean_subtraction:
    #     train_mean = np.mean(train_images, axis=0)
    #     train_images -= train_mean
    #     valid_images -= train_mean
    #     test_images -= train_mean

    # 랜덤 라벨 회전
    # if random_roated_labels:
    #     r, _ = np.linalg.qr(np.random.rand(10, 10))
    #     train_labels = np.dot(train_labels, r)
    #     valid_labels = np.dot(valid_labels, r)
    #     test_labels = np.dot(test_labels, r)

    return (train_images, train_labels, valid_images, valid_labels, test_images, test_labels)
#-----------------------------------------------------------------------------------------------수정

def _select_mnist_subset(datasets,
                         num_train=100,
                         digits=list(range(10)),
                         seed=9999,
                         sort_by_class=False,
                         use_float64=False,
                         mean_subtraction=False,
                         random_roated_labels=False):
  """Select subset of MNIST and apply preprocessing."""
  np.random.seed(seed)
  digits.sort()
  subset = copy.deepcopy(datasets)

  num_class = len(digits)
  num_per_class = num_train // num_class

  idx_list = np.array([], dtype='uint8')

  ys = np.argmax(subset.train.labels, axis=1)  # undo one-hot

  for digit in digits:
    if datasets.train.num_examples == num_train:
      idx_list = np.concatenate((idx_list, np.where(ys == digit)[0]))
    else:
      idx_list = np.concatenate((idx_list,
                                 np.where(ys == digit)[0][:num_per_class]))
  if not sort_by_class:
    np.random.shuffle(idx_list)

  data_precision = np.float64 if use_float64 else np.float32

  train_image = subset.train.images[idx_list][:num_train].astype(data_precision)
  train_label = subset.train.labels[idx_list][:num_train].astype(data_precision)
  valid_image = subset.validation.images.astype(data_precision)
  valid_label = subset.validation.labels.astype(data_precision)
  test_image = subset.test.images.astype(data_precision)
  test_label = subset.test.labels.astype(data_precision)

  if sort_by_class:
    train_idx = np.argsort(np.argmax(train_label, axis=1))
    train_image = train_image[train_idx]
    train_label = train_label[train_idx]

  if mean_subtraction:
    train_image_mean = np.mean(train_image)
    train_label_mean = np.mean(train_label)
    train_image -= train_image_mean
    train_label -= train_label_mean
    valid_image -= train_image_mean
    valid_label -= train_label_mean
    test_image -= train_image_mean
    test_label -= train_label_mean

  if random_roated_labels:
    r, _ = np.linalg.qr(np.random.rand(10, 10))
    train_label = np.dot(train_label, r)
    valid_label = np.dot(valid_label, r)
    test_label = np.dot(test_label, r)

  return (train_image, train_label,
          valid_image, valid_label,
          test_image, test_label)

