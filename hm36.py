import os

import numpy as np

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .custom import CustomDataset
import json
from alphapose.utils.camera import project_point_radial, world_to_camera_frame, load_cameras

# s_hm36_2_coco_jt = [-1, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10, -1]
s_hm36_2_coco_jt = [9, -1, -1, -1, -1, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]

def from_hm36_to_coco_single(pose, num_joints):
    res_jts = np.zeros((num_joints, 2), dtype=np.float)
    # res_vis = np.zeros((num_joints, 2), dtype=np.float)

    for i in range(0, num_joints):
        id1 = i
        id2 = s_hm36_2_coco_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            # res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy()

@DATASET.register_module
class hm36(CustomDataset):
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    visibles =    [2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    num_joints = 17

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        cameras_path = self._root + '/cameras.h5'
        cameras_dict = load_cameras(cameras_path)
        camera_info  = set([c[1] for c in cameras_dict.keys()])
        if self._img_prefix == 'hm36train':
            categories = ['s_01','s_05','s_06','s_07','s_08']
            # categories = ['s_01']
        else:
            categories = ['s_09']
        actions = []
        for i in range(2,16):
            stri = str(i)
            action = 'act_' + stri.zfill(2)
            actions.append(action)
        subact=['subact_01','subact_02']
        cameras=['ca_01','ca_02','ca_03','ca_04']
        items = []
        labels = []

        for cat in categories:
            s = int(cat.split('_')[-1])
            print('loading category ' + cat)
            for act in actions:
                
                for sub in subact:
                    for ca in cameras:
                        ci = int(ca[-1])
                        fname = cat+'_'+act+'_'+sub+'_'+ca
                        annopath = os.path.join(self._ann_file, fname, 'annotation.json')
                        if not os.path.exists(annopath):
                            continue
                        with open(annopath,"r") as f:
                            data = json.load(f)
                        anno = data['annotations']
                        R, T, f, c, k, p, name = cameras_dict[ (s, ci) ]
                        for item in anno:
                            file_name = os.path.join(self._root, 'images', item['file_name'])
                            items.append(file_name)
                            bbox = item['bbox']
                            w_coord_xyz = np.array(item['joint_3d'])
                            width, height = item['width'], item['height']
                            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(bbox), width, height)
                            
                            i_coord_xy, D, radial, tan, r2 =  project_point_radial( w_coord_xyz, R, T, f, c, k, p )
                            i_coord_xy = from_hm36_to_coco_single(i_coord_xy, self.num_joints)
                            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
                            for i in range(self.num_joints):
                                joints_3d[i, 0, 0] = i_coord_xy[i][0]
                                joints_3d[i, 1, 0] = i_coord_xy[i][1]
                                joints_3d[i, :2, 1] = min(self.visibles[i],1)
                            objdict = {
                            'bbox': (xmin, ymin, xmax, ymax),
                            'width': width,
                            'height': height,
                            'joints_3d': joints_3d
                            }
                            labels.append(objdict)

        return items, labels

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
