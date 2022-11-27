train_cfg = dict(model='ecadnet-gc6',
                 epochs=10,
                 batch_size=4,
                 checkpoint_path=rf'../models/kal-ecadnet-gc6/checkpoint.pth',
                 logger_name='ecadnet-gc6',
                 log_path=rf'../models/kal-ecadnet-gc6/train.log',
                 dataset='kalantari',
                 loss_type='l1',
                 two_level_dir=False,
                 use_cpu=False)