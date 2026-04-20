import numpy as np
import torch
import fpsample
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def load_ply(filename):
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
            
            return points
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


def save_ply(filename, pcd):
    """
    点群データを PLY ファイルとして保存する関数。
    出力フォーマットは以下の通り:
      property float32 x
      property float32 y
      property float32 z
      property uint8 r
      property uint8 g
      property uint8 b
      property float32 i
      
    数値の桁指定:
      - x, y, z: 小数点以下4桁まで
      - r, g, b: 整数 (常に 0)
      - i: 小数点以下2桁まで

    例:
      -1334.0197 -1060.7484 1785.6458 0 0 0 0.16

    入力:
      pcd: (N,3) または (N,4) の numpy 配列または torch.Tensor
           4列目が存在する場合は intensity として使用、なければ 0 とする。
    """
    # Noneチェック
    if pcd is None:
        print(f"保存対象が None のためスキップします: {filename}")
        return

    # torch.Tensor の場合は numpy 配列に変換
    if isinstance(pcd, torch.Tensor):
        pcd_np = pcd.cpu().numpy()
    else:
        pcd_np = np.asarray(pcd)

    # 入力の形状チェック (N,3) または (N,4)
    if pcd_np.ndim != 2 or pcd_np.shape[1] not in (3, 4):
        raise ValueError("pcd は (N,3) または (N,4) の形状である必要があります。")

    # 座標は float32 として取得
    xyz = pcd_np[:, :3].astype(np.float32)
    
    # intensity の取得: 4列目があればその値、なければ 0
    if pcd_np.shape[1] == 4:
        intensity = pcd_np[:, 3].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((pcd_np.shape[0], 1), dtype=np.float32)
    
    # r, g, b は uint8 の 0 として生成
    rgb = np.zeros((pcd_np.shape[0], 3), dtype=np.uint8)
    
    # x, y, z, r, g, b, i の順にデータを結合 (shape: (N,7))
    data = np.hstack((xyz, rgb, intensity))

    # PLY ヘッダーの作成
    header = f"""ply
                format ascii 1.0
                element vertex {data.shape[0]}
                property float32 x
                property float32 y
                property float32 z
                property uint8 r
                property uint8 g
                property uint8 b
                property float32 i
                end_header
                """
    # ファイルにヘッダーと各点のデータを書き出す
    with open(filename, "w") as f:
        f.write(header)
        for row in data:
            # 書式: x,y,z は小数点以下4桁、i は小数点以下2桁で出力
            f.write(f"{row[0]:.4f} {row[1]:.4f} {row[2]:.4f} {int(row[3])} {int(row[4])} {int(row[5])} {row[6]:.2f}\n")


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rot_mat, translation):
    # point_cloud は (B, C, L) 形状 (C=3 または C=4)
    # rot_mat は (B, 3, 3)
    # translation は (B, 3)
    
    # 1. 最初の3チャンネル (XYZ) だけをスライス
    xyz = point_cloud[:, :3, :]  # 形状: (B, 3, L)
    
    # 2. 3D座標 (XYZ) にのみ回転と並進を適用
    transformed_xyz = torch.matmul(rot_mat, xyz) + translation.unsqueeze(2) # 形状: (B, 3, L)
    
    # 3. チャンネル数に応じて処理を分岐
    if point_cloud.size(1) == 3:
        # 入力が 3 チャンネルだった場合
        return transformed_xyz
    else:
        # 入力が 4 チャンネル (XYZI) だった場合
        # 4チャンネル目 (Intensity) を取得
        intensity = point_cloud[:, 3:, :] # 形状: (B, 1, L)
        
        # 4. 変換後のXYZ と、元のIntensity を連結(cat)して (B, 4, L) に戻す
        return torch.cat((transformed_xyz, intensity), dim=1)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def downsample_pcd(pointcloud, downsample_point, intensity=True) -> np.ndarray:
    if pointcloud.shape[0] < downsample_point:
        downsample_point = pointcloud.shape[0]
    fps_indices = fpsample.fps_sampling(pointcloud, downsample_point)
    downsampled_pc = pointcloud[fps_indices][:, :4]
    return downsampled_pc


def get_correspondence_matrix(pcd_a, pcd_b, threshold):
    """
    KDTreeを使用して距離閾値内の対応点行列を高速に生成する。
    
    Args:
        pcd_a (torch.Tensor): 点群A [3, N]
        pcd_b (torch.Tensor): 点群B [3, M]
        threshold (float): 対応点とみなす距離の閾値
        
    Returns:
        torch.Tensor: 対応行列 [N, M] (Dense Matrix)
    """
    # Numpyに変換 [N, 3], [M, 3]
    # KDTreeは通常CPUで動作するため、一度Numpyに変換します
    pcd_a_np = pcd_a.transpose(0, 1).cpu().detach().numpy()
    pcd_b_np = pcd_b.transpose(0, 1).cpu().detach().numpy()
    
    n = pcd_a_np.shape[0]
    m = pcd_b_np.shape[0]
    
    # ターゲット点群BでKDTreeを構築
    tree_b = KDTree(pcd_b_np)
    
    # 点群Aの各点について、閾値(threshold)内にある点群Bのインデックスを検索
    # query_ball_pointは各点に対するインデックスのリストを返す
    indices_list = tree_b.query_ball_point(pcd_a_np, r=threshold)
    
    # 出力用の行列を準備（ゼロ行列）
    # ※点数が非常に多い場合（例: 10万点x10万点）、ここでメモリ不足になる可能性があります
    corr_matrix = torch.zeros((n, m))
    
    # 対応があった箇所に1を立てる
    for i, neighbors in enumerate(indices_list):
        if neighbors:
            corr_matrix[i, neighbors] = 1.0
            
    return corr_matrix