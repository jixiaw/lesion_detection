from easydict import EasyDict


cfg = EasyDict()
cfg.INPUT_SHAPE = [128, 96, 128]
cfg.OUTPUT_SHAPE = [x for x in cfg.INPUT_SHAPE]
cfg.DATA_ROOT = '/home/jxw/Downloads/mediastinal_1/annoation/'
cfg.anno_file = '/home/jxw/PycharmProjects/centernet3d/results/annoation_128.json'
# cfg.TEST_DATA_ROOT2 = '/home/jxw/Downloads/jiaotong_test/solitary/'
# cfg.TEST_DATA_ROOT = '/home/jxw/Downloads/jiaotong_test/multi/'
cfg.train_anno_file = '/home/jxw/Downloads/mediastinal_1/annoation/annoation_128.json'
cfg.test_anno_file = '/home/jxw/Downloads/jiaotong_test/annoation_128.json'
cfg.label_file = '/home/jxw/Downloads/mediastinal_1/label1.json'
#cfg.train_results_file = './results/train_single_torch.json'
cfg.train_results_file = './results/detect_5_fold.json'
cfg.train_results_file = './results2/detection.json'
cfg.test_results_file2 = './results/test_single_torch.json'
cfg.test_results_file = './results/test_multi_torch.json'
cfg.crop_data_root = "/media/jxw/软件/linux/data/mediastinal"
cfg.crop_anno_file = "/media/jxw/软件/linux/data/mediastinal/anno_crop.json"
cfg.CHECKPOINTS_ROOT = './checkpoints'
cfg.cross_validation = '/home/jxw/PycharmProjects/centernet3d/results/kflod5_new.json'
cfg.STEPS_PER_EPOCH = 936 // 4
cfg.MAX_KEEPS_CHECKPOINTS = 10
cfg.bboxs_per_scan = 4
cfg.mean = 0.196
cfg.std = 0.278
