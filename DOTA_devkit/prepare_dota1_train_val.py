import os
import os.path as osp

from DOTA_devkit.DOTA2JSON import generate_json_labels
from DOTA_devkit.DOTA2COCO import DOTA2COCOTrain, DOTA2COCOTest, wordname_15

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
from DOTA_devkit.SplitOnlyImage_multi_process import \
    splitbase as splitbase_test


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):
    """Prepare DOTA split data and labels
    Args:
    src_path: dataset path
    dst_path: output path
    gap: overlapping area
    subsize: size of chip image
    scales: multi-scale settings
    num_process: num of processer
    """
    dst_train_path = osp.join(dst_path, 'train_split')
    # dst_val_path = osp.join(dst_path, 'val_split')
    dst_trainval_path = osp.join(dst_path, 'trainval_split')
    dst_test_base_path = osp.join(dst_path, 'test_split')
    dst_test_path = osp.join(dst_path, 'test_split/images')


    # make dst path if not exist
    mkdir_if_not_exists(dst_path)
    mkdir_if_not_exists(dst_train_path)
    # mkdir_if_not_exists(dst_val_path)
    mkdir_if_not_exists(dst_test_base_path)
    mkdir_if_not_exists(dst_test_path)
    # split train data
    print('split train data')
    split_train = splitbase_trainval(osp.join(src_path, 'train'), dst_train_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_train.splitdata(scale)


    # split test data
    print('split test data')
    split_test = splitbase_test(osp.join(src_path, 'val/images'), dst_test_path,
                                gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_test.splitdata(scale)

    mkdir_if_not_exists(dst_trainval_path)
    print('generate labels with json format')
    generate_json_labels(dst_train_path, osp.join(
        dst_train_path, 'trainval.json'))
    generate_json_labels(dst_test_base_path, osp.join(
        dst_test_base_path, 'test.json'), trainval=False)


    print('generate labels with coco format')
    DOTA2COCOTrain(dst_train_path,
                   osp.join(dst_train_path, 'trainval.json'),
                   wordname_15)
    DOTA2COCOTest(dst_test_base_path,
                  osp.join(dst_test_base_path, 'test.json'),
                  wordname_15)


if __name__ == '__main__':
    # single scale
    prepare_multi_scale_data('/mnt/SSD/lwt_workdir/data/dota/',
                             '/mnt/SSD/lwt_workdir/data/dota_1024_train_val',scales=[1.0])
    print('done')
