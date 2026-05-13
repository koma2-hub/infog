import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import traceback

# 既存のユーティリティ関数をインポート（環境に合わせてパスを調整してください）
from utils import load_ply, downsample_pcd, get_correspondence_matrix


class MyLabsDataset(Dataset):
    """
    前処理されたソース点群とターゲット点群を読み込み,2つの点群間の対応点の行列,変換行列を返す
    """
    def __init__(self, dataset_dir, intensity=True):
        self.dataset_dir = dataset_dir
        self.intensity = intensity
        self.file_paths = []
        if os.path.exists(self.dataset_dir):
            for f in os.listdir(self.dataset_dir):
                if f.endswith(".pt"):
                    self.file_paths.append(os.path.join(self.dataset_dir, f))
        self.file_paths.sort()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, weights_only=False)

        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        correspondence_matrix = torch.as_tensor(data['correspondence_matrix']).float()

        if not self.intensity:
            # 輝度情報(4列目以降)を除外
            src_pcd = src_pcd[:3, :]
            tgt_pcd = tgt_pcd[:3, :]

        return src_pcd, tgt_pcd, R_st, t_st, euler_st, correspondence_matrix


def sample_knn_patches_with_overlap(points_full, 
                                    num_points_k=1024, 
                                    overlap_ratio_range=(0.3, 0.5), 
                                    max_retries=50):
    """
    点群から重複率を指定範囲に収めた2つのパッチをサンプリングする
    """
    N, D = points_full.shape
    K = num_points_k
    min_overlap, max_overlap = overlap_ratio_range

    if N < K:
        indices = np.random.choice(N, K, replace=True)
        patch = points_full[indices, :]
        return patch, patch

    coords_xyz = points_full[:, :3]
    tree = KDTree(coords_xyz)

    indices_src = None
    indices_tgt = None

    for _ in range(max_retries):
        center_index_1 = np.random.randint(0, N)
        center_point_1 = coords_xyz[center_index_1]
        _, indices_tgt = tree.query(center_point_1, k=K)

        center_index_2 = np.random.choice(indices_tgt)
        center_point_2 = coords_xyz[center_index_2]
        _, indices_src = tree.query(center_point_2, k=K)

        set_tgt = set(indices_tgt)
        set_src = set(indices_src)
        num_intersection = len(set_tgt.intersection(set_src))
        actual_overlap_ratio = num_intersection / K

        if min_overlap <= actual_overlap_ratio <= max_overlap:
            points_tgt_patch = points_full[indices_tgt, :]
            points_src_patch = points_full[indices_src, :]
            return points_src_patch, points_tgt_patch

    # リトライ上限に達した場合のフォールバック
    idx = np.random.choice(N, K, replace=False)
    p = points_full[idx, :]
    return p, p


def random_rotation(pcd, rotation_range=(0, np.pi/4)):
    """ランダムな回転を生成して適用する"""
    angle_x = np.random.uniform(*rotation_range)
    angle_y = np.random.uniform(*rotation_range)
    angle_z = np.random.uniform(*rotation_range)

    # 変換用のRotationオブジェクト生成 (ZYXオイラー角)
    rotation_st = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])
    rotation_matrix = rotation_st.as_matrix()
    euler = np.asarray([angle_z, angle_y, angle_x])

    pcd_rotated = pcd.copy()
    if pcd.shape[1] >= 3:
        pcd_rotated[:, :3] = rotation_st.apply(pcd[:, :3])
    
    return pcd, pcd_rotated, rotation_matrix, euler


def random_transform(pcd, translation_range=(-5, 5)):
    """ランダムな並進を適用する"""
    translation_vector = np.random.uniform(translation_range[0], translation_range[1], size=3)
    pcd_translated = np.copy(pcd)
    pcd_translated[:, :3] = pcd_translated[:, :3] + translation_vector
    return pcd, pcd_translated, translation_vector


def rigit_transform(pcd):
    """回転と並進を組み合わせて剛体変換を生成する"""
    pcd, pcd_rotated, rotation_matrix, euler_zyx = random_rotation(pcd)
    _, pcd_rotated_transformed, transform_vector = random_transform(pcd_rotated)
    
    rotation_src_to_tgt = rotation_matrix
    rotation_tgt_to_src = rotation_src_to_tgt.T
    transform_tgt_to_src = -rotation_tgt_to_src.dot(transform_vector)
    
    euler_src_to_tgt = euler_zyx
    # 逆回転のオイラー角計算（簡易化のため反転させているが、本来は行列から再計算が正確）
    euler_tgt_to_src = -euler_zyx[::-1]
    
    return pcd, pcd_rotated_transformed, rotation_src_to_tgt, transform_vector, \
           rotation_tgt_to_src, transform_tgt_to_src, euler_src_to_tgt, euler_tgt_to_src


def make_dataset(sample_point, k, data_path, output_dir, overlap_ratio, intensity=True):
    os.makedirs(output_dir, exist_ok=True)

    file_names = [f for f in os.listdir(data_path) if f.endswith('.ply')]
    print(f"処理する点群データ: {len(file_names)}個")

    pair_count = 0
    pbar = tqdm(total=len(file_names), desc="Generating data pairs")

    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)

        if pcd is None:
            pbar.update(1)
            continue

        # ダウンサンプリング
        ds_pcd = downsample_pcd(pcd, sample_point, intensity=True)

        # パッチサンプリング
        src_pcd_orig, tgt_pcd_orig = sample_knn_patches_with_overlap(ds_pcd, num_points_k=k, overlap_ratio_range=overlap_ratio)

        # ターゲットに剛体変換を適用 (Source基準の座標系から移動させる)
        _, transformed_tgt, R_st, translation_st, _, _, euler_st, _ = rigit_transform(tgt_pcd_orig)

        # 点の順番をシャッフル
        src_pcd = src_pcd_orig[np.random.permutation(k)].T # [D, K]
        transformed_tgt = transformed_tgt[np.random.permutation(k)].T # [D, K]

        # --- 対応点行列の計算 ---
        # 修正ポイント: 全てを明示的にTensorに変換し、型を合わせる
        src_pcd_tensor = torch.from_numpy(src_pcd).float()
        R_tensor = torch.from_numpy(R_st).float()
        t_tensor = torch.from_numpy(translation_st).float().view(3, 1)

        # srcをR, tで変換して、transformed_tgtと同じ座標系に持っていく
        # src_pcd_transformed = R * src + t
        src_xyz_transformed = torch.matmul(R_tensor, src_pcd_tensor[:3, :]) + t_tensor

        # 行列計算用にtgtのXYZを抽出
        tgt_xyz_tensor = torch.from_numpy(transformed_tgt[:3, :]).float()

        # 閾値 1.0 で対応行列を取得 (ユーティリティ関数を使用)
        correspondence_matrix = get_correspondence_matrix(src_xyz_transformed, tgt_xyz_tensor, 1.0)

        # データの保存
        data_dict = {
            'src_pcd': src_pcd.astype(np.float32),
            'tgt_pcd': transformed_tgt.astype(np.float32),
            'R_st': R_st.astype(np.float32),
            't_st': translation_st.astype(np.float32),
            'euler_st': euler_st.astype(np.float32),
            'correspondence_matrix': correspondence_matrix.cpu().numpy().astype(np.float32)
        }

        output_filename = os.path.join(output_dir, f"pair_{pair_count:06d}.pt")
        torch.save(data_dict, output_filename)

        pair_count += 1
        pbar.update(1)

    pbar.close()
    print(f"完了: {pair_count}ペアのデータを {output_dir} に書き出し")


def main():
    parser = argparse.ArgumentParser(description='データセットのパラメータ指定')
    parser.add_argument('--data_path', type=str, required=True, help='未処理のデータが格納されているディレクトリのパス')
    parser.add_argument('--output_dir', type=str, required=True, help='データセットの保存先')
    parser.add_argument('--sample_point', type=int, default=8192)
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print("警告: すでにデータセットディレクトリが存在し、ファイルが含まれています。処理を続行しますか？ (y/n)")
        # インタラクティブな確認が必要な場合はここで処理。今回は自動続行。

    make_dataset(sample_point=args.sample_point, k=2048, 
                 data_path=args.data_path, output_dir=args.output_dir, 
                 overlap_ratio=(0.3, 0.5), intensity=True)

    print("データセットの作成完了。読み込みテストを開始します。")

    try:
        MyLabs = MyLabsDataset(args.output_dir, intensity=True)
        if len(MyLabs) == 0:
            print("エラー: データセットが空です。")
            return
            
        print("データセットの総数:", len(MyLabs))
        MyLabs_loader = DataLoader(MyLabs, batch_size=min(8, len(MyLabs)), shuffle=True, num_workers=0)

        batch = next(iter(MyLabs_loader))
        src, tgt, rotation, translation, euler, correspondence = batch

        print("ソース点群の形状:", src.shape)
        print("ターゲット点群の形状:", tgt.shape)
        print("回転行列の形状:", rotation.shape)
        print("並進行列の形状:", translation.shape)
        print("対応点行列の形状:", correspondence.shape)

    except Exception:
        print("データセットのテスト中にエラーが発生しました:")
        traceback.print_exc()


if __name__ == '__main__':
    main()