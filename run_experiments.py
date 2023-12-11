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

"""Run experiments with NNGP Kernel.

'''
python run_experiments.py --num_train=100 --num_eval=10000 --hparams="nonlinearity=tanh,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10
python run_experiments.py --num_train=100 --num_eval=10000 --hparams="nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10
python run_experiments.py --dataset=mnist --hparams="nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10
python run_experiments.py --dataset=mnist --hparams="nonlinearity=tanh,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10


cifar-10
python run_experiments.py --num_train=100 --num_eval=10000 --hparams="nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --dataset=cifar10
python run_experiments.py --num_train=40000 --num_eval=10000 --hparams="nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10 --dataset=cifar10

python run_experiments.py --dataset=cifar10 --hparams="nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83" --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

Usage:

python run_experiments.py \
      --num_train=100 \
      --num_eval=1000 \
      --hparams='nonlinearity=relu,depth=10,weight_var=1.79,bias_var=0.83' \
      --n_gauss=501 --n_var=501 --n_corr=500 --max_gauss=10

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os.path
import time

import numpy as np
import tensorflow as tf

import gpr
import load_dataset
import nngp

#-----------------------------------------------------------------------------------------------수정
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------------------------수정

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('hparams', '',
                    'Comma separated list of name=value hyperparameter pairs to'
                    'override the default setting.')
flags.DEFINE_string('experiment_dir', 'C:\\Users\\user\\iCloudDrive\\git\\nngp',
                    'Directory to put the experiment results.') #------------------수정
flags.DEFINE_string('grid_path', './grid_data',
                    'Directory to put or find the training data.')
flags.DEFINE_integer('num_train', 1000, 'Number of training data.')
flags.DEFINE_integer('num_eval', 1000,
                     'Number of evaluation data. Use 10_000 for full eval')
flags.DEFINE_integer('seed', 1234, 'Random number seed for data shuffling')
flags.DEFINE_boolean('save_kernel', False, 'Save Kernel do disk')
flags.DEFINE_string('dataset', 'mnist','Which dataset to use ["mnist",  "cifar10" ]') #------------------수정
flags.DEFINE_boolean('use_fixed_point_norm', False,
                     'Normalize input variance to fixed point variance')

flags.DEFINE_integer('n_gauss', 501,
                     'Number of gaussian integration grid. Choose odd integer.')
flags.DEFINE_integer('n_var', 501,
                     'Number of variance grid points.')
flags.DEFINE_integer('n_corr', 500,
                     'Number of correlation grid points.')
flags.DEFINE_integer('max_var', 100,
                     'Max value for variance grid.')
flags.DEFINE_integer('max_gauss', 10,
                     'Range for gaussian integration.')

#기본 하이퍼파라미터 설정 함수 : 활성화 함수, 가중치 분산, 편향 분산
def set_default_hparams(): 
  return tf.contrib.training.HParams(
      nonlinearity='tanh', weight_var=1.3, bias_var=0.2, depth=2)

# 모델을 평가하는 함수로, 정확도와 평균 제곱 오차(MSE)를 계산
def do_eval(sess, model, x_data, y_data, save_pred=False):
  """Run evaluation."""

  gp_prediction, stability_eps = model.predict(x_data, sess)

  pred_1 = np.argmax(gp_prediction, axis=1)
  accuracy = np.sum(pred_1 == np.argmax(y_data, axis=1)) / float(len(y_data))
  mse = np.mean(np.mean((gp_prediction - y_data)**2, axis=1))
  pred_norm = np.mean(np.linalg.norm(gp_prediction, axis=1))
  tf.logging.info('Accuracy: %.4f'%accuracy)
  tf.logging.info('MSE: %.8f'%mse)

  if save_pred:
    with tf.gfile.Open(
        os.path.join(FLAGS.experiment_dir, 'gp_prediction_stats.npy'),
        'w') as f:
      np.save(f, gp_prediction)

  return accuracy, mse, pred_norm, stability_eps

# NNGP 커널을 사용하여 실험을 실행
def run_nngp_eval(hparams, run_dir):
  """Runs experiments."""

  tf.gfile.MakeDirs(run_dir)
  # Write hparams to experiment directory.
  with tf.gfile.GFile(run_dir + '/hparams', mode='w') as f:
    f.write(hparams.to_proto().SerializeToString())

  tf.logging.info('Starting job.')
  tf.logging.info('Hyperparameters')
  tf.logging.info('---------------------')
  tf.logging.info(hparams)
  tf.logging.info('---------------------')
  tf.logging.info('Loading data')

  # Get the sets of images and labels for training, validation, and
  # # test on dataset.
  if FLAGS.dataset == 'mnist': 
    (train_image, train_label, valid_image, valid_label, test_image,test_label) = load_dataset.load_mnist(
         num_train=FLAGS.num_train,
         mean_subtraction=True,
         random_roated_labels=False)
  #----------------------------------------------------------------- 변경
  elif FLAGS.dataset == 'cifar10':
    # CIFAR-10 데이터 로드
    (train_images, train_labels, valid_images, valid_labels, test_images, test_labels) = load_dataset.load_cifar10(
        num_train =FLAGS.num_train,
        mean_subtraction=True,
        random_roated_labels=False)

    # 이미지 데이터 평탄화
    train_image = train_images.reshape(train_images.shape[0], -1)
    valid_image = valid_images.reshape(valid_images.shape[0], -1)
    test_image = test_images.reshape(test_images.shape[0], -1)

    # 평탄화된 데이터를 레이블과 함께 사용
    train_label = train_labels
    valid_label = valid_labels
    test_label = test_labels
    #----------------------------------------------------------------- 변경
  else:
    raise NotImplementedError

  tf.logging.info('Building Model')

  if hparams.nonlinearity == 'tanh':
    nonlin_fn = tf.tanh
  elif hparams.nonlinearity == 'relu':
    nonlin_fn = tf.nn.relu
  else:
    raise NotImplementedError

# TensorFlow 세션 내에서 NNGP 커널과 가우시안 프로세스 회귀(GPR) 모델을 설정
  with tf.Session() as sess:
    # Construct NNGP kernel : 지정된 매개변수(깊이, 가중치 분산, 편향 분산 등)를 사용하여 NNGPKernel 클래스의 인스턴스를 생성
    nngp_kernel = nngp.NNGPKernel(
        depth=hparams.depth,
        weight_var=hparams.weight_var,
        bias_var=hparams.bias_var,
        nonlin_fn=nonlin_fn,
        grid_path=FLAGS.grid_path,
        n_gauss=FLAGS.n_gauss,
        n_var=FLAGS.n_var,
        n_corr=FLAGS.n_corr,
        max_gauss=FLAGS.max_gauss,
        max_var=FLAGS.max_var,
        use_fixed_point_norm=FLAGS.use_fixed_point_norm)

    # 가우시안 프로세스 회귀 모델 구성 : 앞서 정의된 NNGP 커널과 훈련 데이터를 사용하여 GPR 모델을 인스턴스화
    model = gpr.GaussianProcessRegression(
        train_image, train_label, kern=nngp_kernel)

    start_time = time.time()
    tf.logging.info('Training')

    # For large number of training points, we do not evaluate on full set to
    # save on training evaluation time.
    # 훈련 데이터셋, 검증 데이터셋, 테스트 데이터셋에 대한 평가 수행
    # 5000개를 초과하는 경우, 처음 1000개의 훈련 데이터만 사용하여 평가
    if FLAGS.num_train <= 50000: #데이터 포인트 너무 많으면 일부만 가지고 함
      acc_train, mse_train, norm_train, final_eps = do_eval(
          sess, model, train_image[:FLAGS.num_eval],
          train_label[:FLAGS.num_eval])
      tf.logging.info('Evaluation of training set (%d examples) took '
                      '%.3f secs'%(
                          min(FLAGS.num_train, FLAGS.num_eval),
                          time.time() - start_time))
    else:
      acc_train, mse_train, norm_train, final_eps = do_eval(
          sess, model, train_image[:1000], train_label[:1000])
      tf.logging.info('Evaluation of training set (%d examples) took '
                      '%.3f secs'%(1000, time.time() - start_time))

    start_time = time.time()
    tf.logging.info('Validation')
    acc_valid, mse_valid, norm_valid, _ = do_eval(
        sess, model, valid_image[:FLAGS.num_eval],
        valid_label[:FLAGS.num_eval])
    tf.logging.info('Evaluation of valid set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))

    start_time = time.time()
    tf.logging.info('Test')
    acc_test, mse_test, norm_test, _ = do_eval(
        sess,
        model,
        test_image[:FLAGS.num_eval],
        test_label[:FLAGS.num_eval],
        save_pred=False)
    tf.logging.info('Evaluation of test set (%d examples) took %.3f secs'%(
        FLAGS.num_eval, time.time() - start_time))
  
  # 결과 기록 : 훈련, 검증, 테스트 데이터셋
  metrics = {
      'train_acc': float(acc_train),
      'train_mse': float(mse_train),
      'train_norm': float(norm_train),
      'valid_acc': float(acc_valid),
      'valid_mse': float(mse_valid),
      'valid_norm': float(norm_valid),
      'test_acc': float(acc_test),
      'test_mse': float(mse_test),
      'test_norm': float(norm_test),
      'stability_eps': float(final_eps),
  }
  # 결과를 csv 파일 저장
  column_names = [
    'num_train', 'nonlinearity', 'weight_var', 'bias_var', 'depth',
    'acc_train', 'acc_valid', 'acc_test', 'mse_train', 'mse_valid',
    'mse_test', 'final_eps'
]
  record_results = [
      FLAGS.num_train, hparams.nonlinearity, hparams.weight_var,
      hparams.bias_var, hparams.depth, acc_train, acc_valid, acc_test,
      mse_train, mse_valid, mse_test, final_eps
  ]
  
  if nngp_kernel.use_fixed_point_norm:
    metrics['var_fixed_point'] = float(nngp_kernel.var_fixed_point_np[0])
    record_results.append(nngp_kernel.var_fixed_point_np[0])

  # Store data
  result_file = os.path.join(run_dir, 'results.csv')
  if not os.path.exists(result_file):
    with tf.gfile.Open(result_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)  # 컬럼명 작성
        writer.writerow(record_results)  # 데이터 작성
  else:
    with tf.gfile.Open(result_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(record_results)  # 데이터만 추가
        
  return metrics


def main(argv):
  del argv  # Unused
  hparams = set_default_hparams().parse(FLAGS.hparams)
  run_nngp_eval(hparams, FLAGS.experiment_dir)


if __name__ == '__main__':
  tf.app.run(main)

