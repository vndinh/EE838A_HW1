import tensorflow as tf
import os
import shutil
from train import training
from valid import validate
from evaluation import evaluate
from config import config

valid_lr_dir = config.VALID.lr_img_path
valid_hr_dir = config.VALID.hr_img_path
valid_sr_gen = config.VALID.sr_gen_path
logs_valid = config.VALID.logs_valid

test_lr_dir = config.TEST.lr_img_path
test_hr_dir = config.TEST.hr_img_path
test_sr_gen = config.TEST.sr_gen_path
logs_test = config.TEST.logs_test

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='train, valid, test')
  args = parser.parse_args()
  if args.mode == 'train':
    training()
  elif args.mode == 'valid':
    validate(valid_lr_dir, valid_hr_dir, valid_sr_gen, logs_valid, is_valid=True)
    evaluate(valid_hr_dir, valid_sr_gen, is_valid=True)
  elif args.mode == 'test':
    validate(test_lr_dir, test_hr_dir, test_sr_gen, logs_test, is_valid=False)
    evaluate(test_hr_dir, test_sr_gen, is_valid=False)
  else:
    raise Exception("Unknown mode")
    