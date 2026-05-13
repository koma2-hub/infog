#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import DCP
from utils import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import random_split 

try:
    from data import MyLabsDataset
except ImportError:
    print("Error: Could not import 'MyLabsDataset' from 'data.py'.")
    exit(1)

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')
    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()
    def close(self):
        self.f.close()

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints/' + args.exp_name + '/main.py.backup')
    os.system('cp model.py checkpoints/' + args.exp_name + '/model.py.backup')

def compute_corr_loss(pred_matrix, gt_matrix):
    """対応点確率行列の負の対数尤度 (確率の正しさ)"""
    eps = 1e-8
    pos_loss = -gt_matrix * torch.log(pred_matrix + eps)
    num_pos = torch.sum(gt_matrix)
    if num_pos > 0:
        loss = torch.sum(pos_loss) / num_pos
    else:
        loss = F.binary_cross_entropy(pred_matrix, gt_matrix)
    return loss

def compute_point_loss(transformed_src, target, gt_matrix):
    """
    点群の重なり誤差 (幾何学的な正しさ)
    正解の対応ペア (gt_matrix == 1) における、移動後のSource点とTarget点の距離を最小化する
    """
    # transformed_src: [B, 3, N], target: [B, 3, N], gt_matrix: [B, N, N]
    batch_size, _, num_points = transformed_src.size()
    
    # [B, 3, N] -> [B, N, 3]
    src_t = transformed_src.transpose(1, 2).contiguous()
    tgt_t = target.transpose(1, 2).contiguous()

    # gt_matrix を用いて、各Source点に対応すべきTarget点の座標を抽出
    # gt_matrix が one-hot でない場合（重複点がある等）も考慮し、行列積で対応座標を求める
    # [B, N, N] @ [B, N, 3] -> [B, N, 3]
    # gt_matrixの各行の和で割り、正規化された対応ターゲット座標を得る
    row_sum = torch.sum(gt_matrix, dim=2, keepdim=True) + 1e-8
    normalized_gt = gt_matrix / row_sum
    target_assigned = torch.matmul(normalized_gt, tgt_t)

    # 実際に正解ペアが存在する点（行の和が0より大きい点）のみを対象にMSEを計算
    mask = (torch.sum(gt_matrix, dim=2) > 0).float()
    diff_sq = torch.sum((src_t - target_assigned) ** 2, dim=2)
    point_loss = torch.sum(diff_sq * mask) / (torch.sum(mask) + 1e-8)
    
    return point_loss

def test_one_epoch(args, net, test_loader):
    net.eval()
    total_loss = 0
    num_examples = 0
    mse_ab = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    eulers_ab = []

    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, euler_ab, correspondence_matrix in tqdm(test_loader):
            src, target = src.cuda(), target.cuda()
            rotation_ab = rotation_ab.cuda()
            translation_ab = translation_ab.cuda().view(-1, 3)
            correspondence_matrix = correspondence_matrix.cuda().float()

            batch_size = src.size(0)
            num_examples += batch_size
            
            rotation_ab_pred, translation_ab_pred, _, _, scores_ab_pred = net(src, target)

            # --- ハイブリッド評価 ---
            corr_loss = compute_corr_loss(scores_ab_pred, correspondence_matrix)
            transformed_src = transform_point_cloud(src[:, :3, :], rotation_ab_pred, translation_ab_pred)
            point_loss = compute_point_loss(transformed_src, target[:, :3, :], correspondence_matrix)
            
            # テスト時は幾何学的整合性を重視したLossを表示
            loss = corr_loss + point_loss
            total_loss += loss.item() * batch_size

            # 統計用
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())

            mse_ab += torch.mean((transformed_src - target[:, :3, :]) ** 2, dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
    eulers_ab = np.concatenate(eulers_ab, axis=0)

    return total_loss / num_examples, mse_ab / num_examples, \
           rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab

def train_one_epoch(args, net, train_loader, opt):
    net.train()
    total_loss = 0
    num_examples = 0

    for src, target, rotation_ab, translation_ab, euler_ab, correspondence_matrix in tqdm(train_loader):
        src, target = src.cuda(), target.cuda() 
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda().view(-1, 3)
        correspondence_matrix = correspondence_matrix.cuda().float()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        
        rotation_ab_pred, translation_ab_pred, _, _, scores_ab_pred = net(src, target)
        
        # --- 安定化されたハイブリッド損失 ---
        # 1. 対応点確率の損失 (NLL)
        corr_loss = compute_corr_loss(scores_ab_pred, correspondence_matrix)
        
        # 2. 幾何学的な点位置の損失 (重なりを強制)
        transformed_src = transform_point_cloud(src[:, :3, :], rotation_ab_pred, translation_ab_pred)
        point_loss = compute_point_loss(transformed_src, target[:, :3, :], correspondence_matrix)
        
        # 3. 変換行列の直接損失 (補助的な制約)
        rot_loss = F.mse_loss(rotation_ab_pred, rotation_ab)
        trans_loss = F.smooth_l1_loss(translation_ab_pred, translation_ab)
        
        # スケールを合わせるための重み付け
        # point_loss が大きい初期段階でも、corr_loss の勾配を殺さないようにバランスをとる
        loss = corr_loss + (point_loss * 0.01) + rot_loss + (trans_loss * 0.1)

        loss.backward()
        # 勾配クリッピング (SVD起因の爆発防止)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        
        total_loss += loss.item() * batch_size

    return total_loss / num_examples

def test(args, net, test_loader, boardio, textio):
    test_loss, test_mse_ab, \
    test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, test_translations_ab_pred, test_eulers_ab = test_one_epoch(args, net, test_loader)
    
    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    gt_eulers_deg = np.degrees(test_eulers_ab)
    diff_euler = (test_rotations_ab_pred_euler - gt_eulers_deg + 180) % 360 - 180
    
    test_r_mse_ab = np.mean(diff_euler ** 2)
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)

    textio.cprint('== FINAL TEST ==')
    textio.cprint('Loss: %f, PC MSE: %f, Rot MSE: %f, Trans MSE: %f' % 
                  (test_loss, test_mse_ab, test_r_mse_ab, test_t_mse_ab))

def train(args, net, train_loader, test_loader, boardio, textio):
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[30, 60], gamma=0.2)

    best_test_mse = np.inf

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(args, net, train_loader, opt)
        test_loss, test_mse_ab, _, _, _, _, _ = test_one_epoch(args, net, test_loader)
        
        # 保存の判断基準を PC MSE (実際の重なり) に置く
        if test_mse_ab <= best_test_mse:
            best_test_mse = test_mse_ab
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint(f'== EPOCH {epoch} ==')
        textio.cprint(f'Loss: [T: {train_loss:.4f}, V: {test_loss:.4f}], PC MSE: {test_mse_ab:.4f}, Best: {best_test_mse:.4f}')

        boardio.add_scalar('train/loss', train_loss, epoch)
        boardio.add_scalar('test/loss', test_loss, epoch)
        boardio.add_scalar('test/pc_mse', test_mse_ab, epoch)
        
        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='DCP Hybrid Loss Training (Correspondence Focus)')
    parser.add_argument('--exp_name', type=str, default='exp_hybrid_geometric', help='Experiment Name')
    parser.add_argument('--model', type=str, default='dcp', choices=['dcp'])
    parser.add_argument('--emb_nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn', 'dgcnnv2'])
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'])
    parser.add_argument('--head', type=str, default='svd', choices=['mlp', 'svd'])
    parser.add_argument('--emb_dims', type=int, default=512)
    parser.add_argument('--n_blocks', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--ff_dims', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_intensity', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_sgd', action='store_true', default=False)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--cycle', type=bool, default=False)

    args = parser.parse_args()
    _init_(args)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    
    all_data_dataset = MyLabsDataset(args.data_path, intensity=args.use_intensity)
    train_size = int(len(all_data_dataset) * 0.8) 
    test_size = len(all_data_dataset) - train_size
    train_dataset, test_dataset = random_split(all_data_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    net = DCP(args).cuda()
    if args.model_path != '':
        net.load_state_dict(torch.load(args.model_path), strict=False)

    if args.eval:
        test(args, net, test_loader, None, textio)
    else:
        boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
        train(args, net, train_loader, test_loader, boardio, textio)
        boardio.close()

if __name__ == '__main__':
    main()