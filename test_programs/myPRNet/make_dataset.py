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
from util import load_ply, downsample_pcd


def calculate_centroid(pcd):
    """(N, D) の点群から重心 (D,) を計算"""
    # 座標 (XYZ) のみで重心を計算
    centroid = np.mean(pcd[:, :3], axis=0)
    return centroid

def get_pcd_around_centroid(pcd, overlap_num):
    # (この関数は現在使用されていないようです)
    centroid = calculate_centroid(pcd)
    distance = (pcd[:,:3]-centroid)**2
    distance = np.sum(distance, axis=1)
    indices = np.argsort(distance)[:overlap_num:]
    return indices

def random_rotation(pcd, rotation_range=(-np.pi/18, np.pi/18)):
    # (この関数は変更ありません)
    #ランダムな回転行列の生成
    angle_x = np.random.uniform(*rotation_range)
    angle_y = np.random.uniform(*rotation_range)
    angle_z = np.random.uniform(*rotation_range)

    sinx = np.sin(angle_x)
    cosx = np.cos(angle_x)
    siny = np.sin(angle_y)
    cosy = np.cos(angle_y)
    sinz = np.sin(angle_z)
    cosz = np.cos(angle_z)

    #各軸の回転行列
    rotation_x = np.array([[1, 0,    0],
                           [0, cosx, -sinx],
                           [0, sinx, cosx]])
    rotation_y = np.array([[cosy, 0, siny],
                           [0,    1, 0],
                           [-siny, 0, cosy]])
    rotation_z = np.array([[cosz, -sinz, 0],
                           [sinz, cosz,  0],
                           [0,    0,     1]])
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    pcd_rotated = pcd.copy()
    #点群が輝度値を持つときは座標のみ変換する
    #点群が輝度値を持たないときはそのまま変換する
    euler = np.asarray([angle_z, angle_y, angle_x])
    rotation_st = Rotation.from_euler('zyx', [angle_z, angle_y, angle_x])

    if(pcd.shape[1] != 3):
        pcd_rotated[:,:3] = rotation_st.apply(pcd[:,:3])
        return pcd, pcd_rotated, rotation_matrix, euler
    else:
        pcd_rotated = pcd_rotated.apply(pcd)
        return pcd, pcd_rotated, rotation_matrix, euler


def random_transform(pcd, translation_range = (-5, 5)):
    translation_vector = np.array([np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1]),
                                   np.random.uniform(translation_range[0], translation_range[1])])
    
    pcd_translated = np.copy(pcd)
    pcd_translated[:,:3] = pcd_translated[:,:3] + translation_vector
    return pcd, pcd_translated, translation_vector

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    distance, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return distance, idx

#点群を可視化する関数
def visualize_pcd(pcd_list, window_name = "Point Cloud"):
    # (変更ありません)
    pcd_o3d_list = []
    for pcd in pcd_list:
        pointcloud = pcd[:,:3]
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pointcloud)
        rgb = [random.uniform(0,1) for i in range(3)]
        pcd_obj.paint_uniform_color(rgb)
        pcd_o3d_list.append(pcd_obj)
    o3d.visualization.draw_geometries(pcd_o3d_list, window_name=window_name)


def visualize_pcd_overlap(pcd1,pcd2,overlap_pcd):
    # (変更ありません)
    pointcloud1 = pcd1[:, :3]
    pointcloud2 = pcd2[:, :3]
    pointcloud3 = overlap_pcd[:, :3]
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d1 = o3d.geometry.PointCloud()
    pcd_o3d1.points = o3d.utility.Vector3dVector(pointcloud1)
    pcd_o3d1.paint_uniform_color([1, 0, 0])
    #open3dのPointCloudオブジェクトを生成
    pcd_o3d2 = o3d.geometry.PointCloud()
    pcd_o3d2.points = o3d.utility.Vector3dVector(pointcloud2)
    pcd_o3d2.paint_uniform_color([0, 0, 1])
    #
    pcd_o3d3 = o3d.geometry.PointCloud()
    pcd_o3d3.points = o3d.utility.Vector3dVector(pointcloud3)
    pcd_o3d3.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd_o3d1, pcd_o3d2, pcd_o3d3],
                                        window_name="Point Cloud")


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
        # (make_prnetDataset で (C, L) 形式で保存されているため、transpose 不要)
        src_pcd = torch.as_tensor(data['src_pcd']).float()
        tgt_pcd = torch.as_tensor(data['tgt_pcd']).float()
        
        R_st = torch.as_tensor(data['R_st']).float().view(3, 3)
        t_ts = torch.as_tensor(data['t_ts']).float().view(1, 3)
        R_ts = torch.as_tensor(data['R_ts']).float().view(3, 3)
        t_st = torch.as_tensor(data['t_st']).float().view(1, 3)
        euler_st = torch.as_tensor(data['euler_st']).float()
        euler_ts = torch.as_tensor(data['euler_ts']).float()

        # 輝度値を除外する場合
        if not self.intensity:
            src_pcd = src_pcd[:3, :] # (C, L) 形式なので、:3 でスライス
            tgt_pcd = tgt_pcd[:3, :]
            
        return src_pcd, tgt_pcd, R_st, t_st, R_ts, t_ts, euler_st, euler_ts


def sample_knn_patches_with_overlap(points_full, num_points_src=1024, num_points_tgt=768):
    """
    1. 全体からKNNで Src(1024点) を抽出
    2. 抽出したSrcの中から、さらにKNNで Tgt(768点) を抽出
    """
    N, D = points_full.shape

    # 点数が足りない場合のフォールバック
    if N < num_points_src:
        indices_src = np.random.choice(N, num_points_src, replace=True)
        patch_src = points_full[indices_src, :]
        
        indices_sub = np.random.choice(num_points_src, num_points_tgt, replace=False)
        patch_tgt = patch_src[indices_sub, :]
        return patch_src, patch_tgt

    # --- 1. Srcパッチの抽出 (全体からKNN 1024点) ---
    coords_xyz = points_full[:, :3]
    tree_full = KDTree(coords_xyz)

    center_idx_src = np.random.randint(0, N)
    _, indices_src = tree_full.query(coords_xyz[center_idx_src], k=num_points_src)
    
    patch_src = points_full[indices_src, :] # (1024, D)

    # --- 2. Tgtパッチの抽出 (Srcの中からKNN 768点) ---
    # Srcパッチ内の座標で再度KDTreeを構築
    coords_src_xyz = patch_src[:, :3]
    tree_src = KDTree(coords_src_xyz)
    
    # Srcパッチ内の点からランダムに中心を選ぶ
    center_idx_tgt_in_src = np.random.randint(0, num_points_src)
    
    # Src内でのKNNで768点を取得
    _, indices_tgt_in_src = tree_src.query(coords_src_xyz[center_idx_tgt_in_src], k=num_points_tgt)
    
    patch_tgt = patch_src[indices_tgt_in_src, :] # (768, D)

    return patch_src, patch_tgt


# ==========================================
# ★ 変更した関数: make_prnetDataset (呼び出し部分のみ修正)
# ==========================================
def make_prnetDataset(sample_point, k, overlap_ratio, data_path, output_dir, intensity=True):
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_names = os.listdir(data_path)
    print(f"対象ファイル: {file_names}")
    
    pair_counter = 0

    total_pairs_to_generate = len(file_names)
    pbar = tqdm(total=total_pairs_to_generate, desc="Generating data pairs")

    for file in file_names:
        file_path = os.path.join(data_path, file)
        pcd = load_ply(file_path)
        
        if pcd is None: 
            pbar.update(1) # 1ファイルあたり10ペアなのでスキップ数も合わせる
            continue
            
        ds_pcd = downsample_pcd(pcd, sample_point)
        # サンプリングした点群の座標のスケーリング
        ds_pcd[:, :3] = ds_pcd[:, :3] / 124
        
        
        # ★ 変更: ここで新しい引数で関数を呼び出す
        # src=1024, tgt=768 を指定
        src_pcd, tgt_pcd = sample_knn_patches_with_overlap(
            ds_pcd, 
            num_points_src=1024, 
            num_points_tgt=768
        )

        # 3. tgtにランダムな変換をする
        _, transformed_tgt, R_st, translation_st, \
        R_ts, translation_ts, euler_st, euler_ts = rigit_transform(tgt_pcd)
        
        # permutation (シャッフル)
        src_pcd = np.random.permutation(src_pcd)
        transformed_tgt = np.random.permutation(transformed_tgt)
        
        # 4. (N, D) -> (D, N) に転置して保存
        src_pcd = src_pcd.T
        transformed_tgt = transformed_tgt.T
        
        data_dict = {
            'src_pcd': src_pcd.astype(np.float32),       # (D, 1024)
            'tgt_pcd': transformed_tgt.astype(np.float32), # (D, 768)
            'R_st': R_st.astype(np.float32),
            't_st': translation_st.astype(np.float32),
            'R_ts': R_ts.astype(np.float32),
            't_ts': translation_ts.astype(np.float32),
            'euler_st': euler_st.astype(np.float32),
            'euler_ts': euler_ts.astype(np.float32)
        }
        
        output_filename = os.path.join(output_dir, f"pair_{pair_counter:06d}.pt")
        
        torch.save(data_dict, output_filename)
        
        pair_counter += 1
        
        pbar.update(1)

    pbar.close()
    print(f"完了: 合計 {pair_counter} ペアのデータを {output_dir} に書き出しました。")


def test_function(pcd):
    # (変更ありません)
    pcd, pcd_translated, t = random_transform(pcd)
    pcd, pcd_rotated, rotation_matrix ,_ = random_rotation(pcd)
    print('translation', t)
    print('rotation', rotation_matrix)

    visualize_pcd([pcd,pcd_translated])
    visualize_pcd([pcd, pcd_rotated])
    pcd_registered = rotation_matrix.dot(pcd[:,:3]) - t
    visualize_pcd([pcd_registered, pcd_rotated])


def rigit_transform(pcd):
    # (この関数は random_rotation と random_transform を呼ぶだけなので変更不要)
    pcd ,pcd_rotated, rotation_matrix, euler_zyx = random_rotation(pcd)
    _, pcd_rotated_tranfromed , transform_vector = random_transform(pcd_rotated)
    rotation_src_to_tgt = rotation_matrix
    rotation_tgt_to_src = rotation_src_to_tgt.T
    transform_tgt_to_src = -rotation_tgt_to_src.dot(transform_vector)
    euler_src_to_tgt = euler_zyx
    euler_tgt_to_src = -euler_zyx[::-1]
    return pcd, pcd_rotated_tranfromed, rotation_src_to_tgt,transform_vector,\
           rotation_tgt_to_src, transform_tgt_to_src, euler_src_to_tgt, euler_tgt_to_src

def main():
    # --- メイン実行部 ---
    # (実行パスを修正)
    path = "/home/koma2/infog/data/processed"
    output_dir = "/home/koma2/infog/dataset/myprnet" 
    overlap_range = (0.5, 0.7)

    # (1) データセットの再生成
    print(f"データセットを {output_dir} に生成します...")
    make_prnetDataset(
        sample_point=8192, 
        k=1024, 
        overlap_ratio=overlap_range, 
        data_path=path,
        output_dir=output_dir,
        intensity=True # ★ intensity=True を渡すように修正
    )
    print("データセット生成完了。")

    # (2) 生成されたデータセットの読み込みテスト
    print("\n--- データセット読み込みテスト ---")
    processed_path = output_dir

    try:
        train_dataset = PRNetDataset(processed_path, intensity=True)
        print(f"データセット準備完了。合計ペア数: {len(train_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0 # ★ main.py 以外で num_workers > 0 を使うとエラーになることがあるため 0 に変更
        )
        
        batch = next(iter(train_loader))

        src_pcd_batch, tgt_pcd_batch, R_st_batch, t_st_batch, \
        R_ts_batch, t_ts_batch, euler_st_batch, euler_ts_batch = batch
        
        print(f"\n--- 最初のバッチ ---")
        print(f"ソース点群 (Torch Tensor) の形状: {src_pcd_batch.shape}")
        print(f"ターゲット点群 (Torch Tensor) の形状: {tgt_pcd_batch.shape}")
        print(f"変換行列 (Torch Tensor) の形状: {R_st_batch.shape}")
        print(f"並進 (Torch Tensor) の形状: {t_st_batch.shape}")
        
        # ★ 正規化された並進ベクトルの値を確認
        print(f"並進ベクトルのサンプル値 (t_st):\n {t_st_batch[:2]}")

    except Exception as e:
        print(f"データセットのテスト中にエラーが発生しました: {e}")
        print("トレースバック:")
        import traceback
        traceback.print_exc()




if __name__ == '__main__':
    main()
