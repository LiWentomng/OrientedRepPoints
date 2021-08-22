import os
import os.path as osp

from DOTA_devkit.ResultMerge_multi_process import mergebypoly, mergebyrec

dst_path = '/mnt/SSD/lwt_workdir/orientedreppoints/work_dirs/dota_merge/'
dst_raw_path = osp.join(dst_path, 'result_raw')
dst_merge_path = osp.join(dst_path, 'result_merge')

if not osp.exists(dst_merge_path):
    os.mkdir(dst_merge_path)

print('merge dota result')
mergebypoly(dst_raw_path, dst_merge_path)
print('done!')
