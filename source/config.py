from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.IMG = edict()
config.TEST = edict()

# Image Parameters
config.IMG.scale = 2

# Hyper Parameters
config.TRAIN.n_epoch = 100
config.TRAIN.patch_size = 64
config.TRAIN.batch_size = 16
config.TRAIN.learning_rate_init = 0.0001
config.TRAIN.learning_rate_decay = 0.5
config.TRAIN.decay_period = int(config.TRAIN.n_epoch/10)

# Adam Parameters
config.TRAIN.beta1 = 0.7
config.TRAIN.beta2 = 0.3

config.TRAIN.model_dir = '..\\model\\model.ckpt'
config.TRAIN.logs_path = '..\\logs'
config.TRAIN.logs_train = '..\\logs\\logs_train.txt'

config.TRAIN.hr_img_path = '..\\data\\mscoco2017_train_crop\\HR'
config.TRAIN.lr_img_path = '..\\data\\mscoco2017_train_crop\\LR'

config.VALID.hr_img_path = '..\\data\\mscoco2017_val_crop\\HR'
config.VALID.lr_img_path = '..\\data\\mscoco2017_val_crop\\LR'
config.VALID.sr_gen_path = '..\\report\\valid_sr_gen'
config.VALID.logs_valid = '..\\logs\\logs_valid.txt'

config.TEST.hr_img_path = '..\\data\\test\\HR'
config.TEST.lr_img_path = '..\\data\\test\\LR'
config.TEST.logs_test = '..\\logs\\logs_test.txt'
config.TEST.result_path = '..\\report'
config.TEST.sr_gen_path = '..\\report\\test_sr_gen'

