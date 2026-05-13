#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_pt_file(file_path):
    print(f"\n--- 可視化開始: {os.path.basename(file_path)} ---")
    
    # .pt ファイルの読み込み (CPU上にロード)
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # データの取り出し
    # 保存時の形状は [D, K] (Dは次元3 or 4, Kは点数)
    src_pcd_np = data['src_pcd']  
    tgt_pcd_np = data['tgt_pcd']  
    R_st = data['R_st']           # [3, 3]
    t_st = data['t_st'].reshape(3, 1) # [3, 1] に整形
    
    # Open3Dで表示するため、座標部分(XYZ)のみを取り出して [K, 3] に転置
    src_xyz = src_pcd_np[:3, :].T
    tgt_xyz = tgt_pcd_np[:3, :].T
    
    # ==========================================
    # 1. 初期状態（位置合わせ前）の作成と表示
    # ==========================================
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src_xyz)
    src_o3d.paint_uniform_color([1, 0, 0])  # Sourceは赤
    
    tgt_o3d = o3d.geometry.PointCloud()
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt_xyz)
    tgt_o3d.paint_uniform_color([0, 0, 1])  # Targetは青
    
    print("[1/3] 初期状態（位置合わせ前）を表示します。")
    print("  - 赤: ソース点群 (Source)")
    print("  - 青: ターゲット点群 (Target)")
    print("  ※ 3Dウィンドウを閉じるか 'Q' キーを押すと次へ進みます。")
    o3d.visualization.draw_geometries([src_o3d, tgt_o3d], window_name="1. Initial State (Source: Red, Target: Blue)")
    
    # ==========================================
    # 2. グラウンドトゥルース（正解）適用後の作成と表示
    # ==========================================
    # 正解の変換行列（GT）をSourceに適用
    # src_aligned = R * src + t
    src_aligned_xyz = (np.matmul(R_st, src_pcd_np[:3, :]) + t_st).T
    
    src_aligned_o3d = o3d.geometry.PointCloud()
    src_aligned_o3d.points = o3d.utility.Vector3dVector(src_aligned_xyz)
    src_aligned_o3d.paint_uniform_color([1, 0.5, 0])  # 変換後のSourceはオレンジ
    
    print("[2/3] 正解の変換行列(GT)を適用した状態を表示します。")
    print("  - オレンジ: 位置合わせ後のソース点群 (Aligned Source)")
    print("  - 青: ターゲット点群 (Target)")
    o3d.visualization.draw_geometries([src_aligned_o3d, tgt_o3d], window_name="2. Aligned State (Aligned: Orange, Target: Blue)")

    # ==========================================
    # 3. 輝度値(Intensity)がある場合の表示
    # ==========================================
    if src_pcd_np.shape[0] >= 4:
        print("[3/3] 輝度値（Intensity）による色付け表示を行います。")
        src_intensity = src_pcd_np[3, :]
        tgt_intensity = tgt_pcd_np[3, :]
        
        # 輝度値を0.0 ~ 1.0に正規化する関数
        def normalize(i_array):
            i_min, i_max = i_array.min(), i_array.max()
            if i_max - i_min < 1e-6:
                return np.zeros_like(i_array)
            return (i_array - i_min) / (i_max - i_min)
            
        # Matplotlibのカラーマップを利用してスカラー値をRGBに変換
        # ソースとターゲットで別のカラーマップを使用し、重なりを見やすくする
        src_colors = plt.get_cmap('viridis')(normalize(src_intensity))[:, :3] # 青〜黄
        tgt_colors = plt.get_cmap('magma')(normalize(tgt_intensity))[:, :3]   # 黒〜赤〜白
        
        src_aligned_o3d.colors = o3d.utility.Vector3dVector(src_colors)
        tgt_o3d.colors = o3d.utility.Vector3dVector(tgt_colors)
        
        o3d.visualization.draw_geometries([src_aligned_o3d, tgt_o3d], window_name="3. Intensity View (Aligned)")
    else:
        print("[3/3] スキップ: このデータには輝度値（Intensity）が含まれていません。")

def main():
    parser = argparse.ArgumentParser(description='DCP Dataset (.pt) Visualizer')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='作成された .pt ファイル、またはファイル群が含まれるディレクトリのパス')
    parser.add_argument('--num_samples', type=int, default=1, 
                        help='ディレクトリを指定した場合に連続して可視化するファイルの数（デフォルト: 1）')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"エラー: 指定されたパスが存在しません -> {args.data_path}")
        return

    # 単一ファイルが指定された場合
    if os.path.isfile(args.data_path):
        if args.data_path.endswith('.pt'):
            visualize_pt_file(args.data_path)
        else:
            print("エラー: .pt ファイルを指定してください。")
            
    # ディレクトリが指定された場合
    elif os.path.isdir(args.data_path):
        pt_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.pt')]
        pt_files.sort()
        
        if len(pt_files) == 0:
            print(f"エラー: ディレクトリ内に .pt ファイルが見つかりません -> {args.data_path}")
            return
            
        print(f"ディレクトリ内に {len(pt_files)} 個のデータペアが見つかりました。")
        samples_to_show = min(args.num_samples, len(pt_files))
        
        for i in range(samples_to_show):
            visualize_pt_file(pt_files[i])
    else:
        print("エラー: 無効なパスです。")

if __name__ == '__main__':
    main()