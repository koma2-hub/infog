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

# ★ util.py からインポート
from util import load_ply, downsample_pcd

def apply_random_rigid_transform(pcd):
    """
    論文準拠: ランダムな剛体変換
    - 回転: 各軸 [0, 45度] (つまり [0, pi/4])
    - 平行移動: [-0.5, 0.5]
    """
    angle_x = np.random.uniform(0, np.pi/4)
    angle_y = np.random.uniform(0, np.pi/4)
    angle_z = np.random.uniform(0, np.pi/4)
    euler_zyx = np.array([angle_z, angle_y, angle_x])
    
    rotation_st = Rotation.from_euler('zyx', euler_zyx)
    R_st = rotation_st.as_matrix()

    t_st = np.random.uniform(-0.5, 0.5, size=(3,))

    pcd_transformed = pcd.copy()
    pcd_transformed[:, :3] = rotation_st.apply(pcd[:, :3]) + t_st

    return pcd_transformed, R_st, t_st, euler_zyx

def generate_partial_scans(src_full, tgt_full, num_points=768):
    """
    論文準拠: 部分的(Partial)なスキャンのシミュレーション
    空間上のランダムな1点から、それぞれの全体点群に対してKNNで768点を抽出します。
    """
    min_xyz = np.min(src_full[:, :3], axis=0)
    max_xyz = np.max(src_full[:, :3], axis=0)
    
    viewpoint = np.array([
        np.random.uniform(min_xyz[0], max_xyz[0]),
        np.random.uniform(min_xyz[1], max_xyz[1]),
        np.random.uniform(min_xyz[2], max_xyz[2])
    ])

    tree_src = KDTree(src_full[:, :3])
    _, src_indices = tree_src.query(viewpoint, k=num_points)
    src_partial = src_full[src_indices, :]

    tree_tgt = KDTree(tgt_full[:, :3])
    _, tgt_indices = tree_tgt.query(viewpoint, k=num_points)
    tgt_partial = tgt_full[tgt_indices, :]

    return src_partial, tgt_partial

def make_prnetDataset(sample_point, data_path, output_dir, intensity=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # data_path 内の .ply ファイルをすべて取得
    file_names = [f for f in os.listdir(data_path) if f.endswith('.ply')]
    print(f"対象ファイル数: {len(file_names)}")
    
    pair_counter = 0
    pbar = tqdm(total=len(file_names), desc="Generating data pairs")

    for file in file_names:
        file_path = os.path.join(data_path, file)
        
        # 1. データの読み込み (util.py の load_ply を使用)
        pcd = load_ply(file_path, intensity=intensity)
        if pcd is None: 
            pbar.update(1)
            continue
            
        # 2. プレダウンサンプリング (計算量削減のため)
        # 点群が非常に大きい場合、一度指定した sample_point (例: 8192) まで減らす
        if pcd.shape[0] > sample_point:
             pcd = downsample_pcd(pcd, sample_point, intensity=intensity)
             
        # 3. 論文準拠: さらに FPS で全体を1024点にサンプリング (Xの作成)
        # ここでも util.py の downsample_pcd (内部で fpsample を使用) を活用
        src_full = downsample_pcd(pcd, 1024, intensity=intensity)
        
        # 座標のスケーリング
        src_full[:, :3] = src_full[:, :3] / 124.0
        
        # 4. 論文準拠: 全体点群に剛体変換を適用 (Yの作成)
        tgt_full, R_st, t_st, euler_st = apply_random_rigid_transform(src_full)
        
        # 逆変換の計算
        R_ts = R_st.T
        t_ts = -R_ts.dot(t_st)
        euler_ts = -euler_st[::-1]

        # 5. 論文準拠: 同じ視点から768点ずつ抽出 (Partial化)
        src_partial, tgt_partial = generate_partial_scans(src_full, tgt_full, num_points=768)
        
        # 6. 点の順番をシャッフル
        src_partial = np.random.permutation(src_partial)
        tgt_partial = np.random.permutation(tgt_partial)
        
        # 7. (N, D) -> (D, N) に転置して保存
        src_partial = src_partial.T
        tgt_partial = tgt_partial.T
        
        data_dict = {
            'src_pcd': src_partial.astype(np.float32),      # (D, 768)
            'tgt_pcd': tgt_partial.astype(np.float32),      # (D, 768)
            'R_st': R_st.astype(np.float32),
            't_st': t_st.astype(np.float32),
            'R_ts': R_ts.astype(np.float32),
            't_ts': t_ts.astype(np.float32),
            'euler_st': euler_st.astype(np.float32),
            'euler_ts': euler_ts.astype(np.float32)
        }
        
        output_filename = os.path.join(output_dir, f"pair_{pair_counter:06d}.pt")
        torch.save(data_dict, output_filename)
        
        pair_counter += 1
        pbar.update(1)

    pbar.close()
    print(f"完了: 合計 {pair_counter} ペアのデータを {output_dir} に書き出しました。")


class PRNetDataset(Dataset):
    def __init__(self, processed_dir, intensity=True):
        self.processed_dir = processed_dir
        self.intensity = intensity
        
        self.file_paths = sorted([
            os.path.join(processed_dir, f) 
            for f in os.listdir(self.processed_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, weights_only=False) 

        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合 (D, N 形式なので :3 でスライス)
        if not self.intensity:
            src_pcd = src_pcd[:3, :]
            tgt_pcd = tgt_pcd[:3, :]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts


def main():
    # ご自身の環境のパスに設定してください
    path = "/home/koma2/infog/data/processed"
    output_dir = "/home/koma2/infog/dataset/myprnet" 

    print(f"データセットを {output_dir} に生成します...")
    make_prnetDataset(
        sample_point=8192,  # 初期読み込み時の軽量化用
        data_path=path,
        output_dir=output_dir,
        intensity=True      # 輝度値を含める
    )

    print("\n--- データセット読み込みテスト ---")
    try:
        train_dataset = PRNetDataset(output_dir, intensity=True)
        print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        batch = next(iter(train_loader))
        src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
        R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
        
        print(f"\n--- 最初のバッチ ---")
        print(f"ソース点群 の形状 (Batch, C, Points): {src_pcd_batch.shape}")
        print(f"ターゲット点群 の形状 (Batch, C, Points): {tgt_pcd_batch.shape}")
        print(f"変換行列 (R_st) の形状: {R_st_batch.shape}")
        print(f"並進 (t_st) の形状: {t_st_batch.shape}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()