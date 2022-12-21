import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D_dt
#from networks.unet_3D_dv_semi import unet_3D_dv_semi
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='BraTs2019_UGMCL_50', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D_dt', help='model_name')


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/BraTs2019_DTC_25/{}_Prediction".format(
        FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D_dt(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)

