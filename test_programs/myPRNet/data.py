import sys
import os
import numpy as np
import random
import torch 
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import open3d as o3d
from tqdm import tqdm

class PRNetDataset(Dataset):
    """
    前処理済みの .pt ファイルを読み込むデータセットクラス
    """
    def __init__(self, processed_dir, intensity=True, n_points=1024):
        self.processed_dir = processed_dir
        self.intensity = intensity
        self.n_points = n_points # n_points を受け取る (main.py から渡されるため)
        
        # processed_dir にある .pt ファイルのリストを作成
        self.file_paths = []
        for f in os.listdir(self.processed_dir):
            if f.endswith(".pt"):
                self.file_paths.append(os.path.join(self.processed_dir, f))
        
        self.file_paths.sort() # 順序を保証

    def __len__(self):
        # データセットの総数 (ファイルの数) を返す
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # weights_only=False は pickle 化されたNumpy配列を読み込むために必要
        data = torch.load(file_path, weights_only=False) 

        # 3. 辞書からテンソルを取得
        # (make_dcpDataset で (C, L) 形式で保存されているため、transpose 不要)
        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合
        if not self.intensity:
            src_pcd = src_pcd[:3, :] # (C, L) 形式なので、:3 でスライス
            tgt_pcd = tgt_pcd[:3, :]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts

if __name__ == '__main__':
    # デバッグ用コード: データセットの中身を検証する
    import argparse

    print("=== データセット診断モードを開始します ===")

    parser = argparse.ArgumentParser()
    # デフォルト値はエラーログにあったパスを入れていますが、必要に応じて変更してください
    parser.add_argument('--data_path', type=str, 
                        default='/mnt/d/SICK/pay-10-bucks/myDCP/datasetv4', 
                        help='検証するデータセットのパス')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"エラー: 指定されたパスが存在しません -> {args.data_path}")
        sys.exit(1)

    # intensity=True (強度あり) としてロードを試みる
    # もしデータに強度がなければ、ファイルそのまま(3ch)で返ってくるはずです
    try:
        dataset = PRNetDataset(args.data_path, intensity=True)
    except Exception as e:
        print(f"データセットの初期化に失敗しました: {e}")
        sys.exit(1)

    print(f"データパス: {args.data_path}")
    print(f"データ総数: {len(dataset)}")

    if len(dataset) > 0:
        # 最初のデータを取得して形状を確認
        # dataset[0] は (src, tgt, R_st, t_st, R_ts, t_ts, euler_st, euler_ts) を返します
        first_data = dataset[0]
        src_pcd = first_data[0] # 最初の要素が src_pcd

        print("-" * 40)
        print(f"データの形状 (Shape): {src_pcd.shape}")
        
        # 形状チェック (通常 (Channels, Points) の形式)
        num_channels = src_pcd.shape[0]

        if num_channels == 3:
            print("【判定結果】: ⚠️ データは [XYZ (3次元)] のみです。")
            print("  -> 強度(Intensity)情報は含まれていません。")
            print("  -> --use_intensity を使う場合は、データローダーでの0埋め処理が必要です。")
        elif num_channels == 4:
            print("【判定結果】: ✅ データは [XYZ + 強度 (4次元)] です。")
            print("  -> 強度情報が含まれています。")
        else:
            print(f"【判定結果】: ❓ 想定外のチャンネル数です ({num_channels} ch)。")
        print("-" * 40)
    else:
        print("データセットの中身が空です。")