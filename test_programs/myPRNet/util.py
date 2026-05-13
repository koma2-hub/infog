#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation

import fpsample


def euler2mat(angle):
    """
    convert euler angles [B, 3] to rotation matrix, reference: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    :param angle: rx, ry, rz [B, 3]
    :return: rotation matrix [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """
    convert quaternion to rotation matrix ([x, y, z, w] to follow scipy
    :param quat: four quaternion of rotation
    :return: rotation matrix [B, 3, 3]
    """
    # norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    # w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    :param vec: tx, ty, tz, rx, ry, rz [B, 6]
    :param rotation_mode: 'euler' or 'quat'
    :return: rotation matrix [B, 3, 3] and translation matrix [B, 3, 1]
    """
    translation_mat = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    return rot_mat, translation_mat



def mat2euler(rot_mat, seq='xyz'):
    """
    convert rotation matrix to euler angle
    :param rot_mat: rotation matrix rx*ry*rz [B, 3, 3]
    :param seq: seq is xyz(rotate along z first) or zyx
    :return: three angles, x, y, z
    """
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]
    if seq == 'xyz':
        z = torch.atan2(-r12, r11)
        y = torch.asin(r13)
        x = torch.atan2(-r23, r33)
    else:
        y = torch.asin(-r31)
        x = torch.atan2(r32, r33)
        z = torch.atan2(r21, r11)
    return torch.stack((x, y, z), dim=1)


def mat2quat(rot_mat, seq='xyz'):
    """
    covert rotation matrix to quaternion
    :param rot_mat: rotation matrix [B, 3, 3]
    :param seq: 'xyz'(rotate along z first) or 'zyx'
    :return: quaternion of the first three entries
    """
    pass


def mat2pose_vec(rot_mat, translation_mat, rotation_mode='euler', seq='xyz'):
    """
    Convert rotation matrix and translation matrix to 6DoF
    :param rot_mat: [B, 3, 3]
    :param translation_mat: [B, 3, 1]
    :param rotation_mode: 'euler' or quat'
    :param seq: 'xyz'(rotate along z first) or 'zyx'
    :return: pose_vec  - tx, ty, tz, rx, ry, rz [B, 6]
    """
    pass

# 既存の transform_point_cloud をこれに置き換えてください
def transform_point_cloud(point_cloud, rotation, translation):
    """
    point_cloud: [Batch, Channels, N] (Channels >= 3)
    rotation: [Batch, 3, 3]
    translation: [Batch, 3]
    """
    # 1. XYZ座標(最初の3次元)と、その他の特徴量(強度など)に分離する
    xyz = point_cloud[:, :3, :]
    features = None
    if point_cloud.shape[1] > 3:
        features = point_cloud[:, 3:, :]  # 4チャンネル目以降を退避

    # 2. XYZ座標のみを回転・平行移動する
    # translationの形状合わせ: [B, 3] -> [B, 3, 1]
    transformed_xyz = torch.matmul(rotation, xyz) + translation.unsqueeze(2)

    # 3. 退避していた特徴量を再結合する
    if features is not None:
        return torch.cat([transformed_xyz, features], dim=1)
    else:
        return transformed_xyz

# (他の関数 npmat2euler, quat2mat などはそのまま維持してください)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def downsample_pcd(pointcloud, downsample_point, intensity=False) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud[:,:3], downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc

def load_ply(filename, intensity=True):
    #.plyファイルを読み込み　点群(x, y, z, intensity)のnumpy配列を返す
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            header_index = None
            for i, line in enumerate(lines):
                if 'end_header' in line:
                    header_index = i
                    break
            if header_index is None:
                raise ValueError("PLYファイルのヘッダが正しく読み込めませんでした。")
            
            # ヘッダ以降の行を読み込み
            points = np.array([list(map(float, l.split())) for l in lines[header_index+1:]])
            if points.shape[1] < 4:
                # xyz + intensity(もしくは他の属性)がなければエラー
                raise ValueError(f"期待される列数に満たないデータが検出されました: {points.shape[1]}列")
            
            # [x, y, z, intensity] の形に整形
            # intensity が最後の列にあると仮定 (points[:, -1])
            points = np.concatenate([points[:, :3], points[:, -1].reshape(-1, 1)], axis=1)
            if(intensity):
                return points
            else:
                return points[:, :3]
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
        # 必要に応じて sys.exit(1) などで終了するか、Noneを返す
        return None
    except ValueError as e:
        print(f"PLYファイルの読み込みエラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました(load_ply): {e}")
        return None
    

def knn(x: np.ndarray, k: int):
    """
    一つの点群データ `x` の各点について、k近傍点のインデックスをNumPyで計算する。

    Args:
        x: 点群データ。形状は (N, C)。
           N: 点の数
           C: 特徴量次元
        k: 探す近傍点の数。

    Returns:
        np.ndarray: 各点のk近傍点のインデックス。形状は (N, k)。
    """
    num_points = x.shape[0]
    x = x[:,:3]

    x_norm_sq = np.sum(x**2, axis=1, keepdims=True)  # 形状: (N, 1)
    dot_product = np.matmul(x, x.T)  # 形状: (N, N)
    dist_matrix = x_norm_sq - 2 * dot_product + x_norm_sq.T  # 形状: (N, N)
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]

    return indices


if __name__ == '__main__':
    print('hello world')
