import os
import glob
import logging
import numpy as np
import torch
import pyvista as pv
import trimesh
import trimesh.transformations as tf
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_scipy
from natsort import natsorted

class BreastPhantomDataset(Dataset):
    def __init__(self, cfg, phase='train', transform=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.phase = phase
        self.cfg = cfg
        self.transform = transform
        
        # --- 1. PATH SETUP ---
        # root/train/
        self.phase_dir = os.path.join(cfg.root, phase) # train or val
        # root/train/deformed/
        self.deformed_dir = os.path.join(self.phase_dir, 'deformed')
        
        if not os.path.exists(self.deformed_dir):
             raise ValueError(f"Deformed data directory not found: {self.deformed_dir}")

        self.und_path = os.path.join(self.phase_dir, "undeformed.npz")
        
        if not os.path.exists(self.und_path):
            raise FileNotFoundError(f"Undeformed file not found at: {self.und_path}")

        # Locate all Deformed files
        self.deformed_paths = sorted(glob.glob(os.path.join(self.deformed_dir, "*.npz")))
        self.logger.info(f'Found {len(self.deformed_paths)} deformed samples matching 1 undeformed source.')

        # --- 2. PRE-LOAD UNDEFORMED MESH (Optimization) ---
        # Since this is shared by ALL samples, we load/subdivide it once here.
        self.logger.info("Pre-loading and subdividing source mesh...")
        data_und = np.load(self.und_path)
        tm_und = trimesh.Trimesh(vertices=data_und['vertices'], faces=data_und['faces'])
        dense_und = tm_und.subdivide() # Subdivide once
        
        # Store in memory as float32 for PyTorch
        self.verts_und = dense_und.vertices.astype(np.float32)
        
        # Pre-calculate ROI indices
        roi_mask = self.verts_und[:, 1] < 25.0
        self.valid_roi_indices = np.where(roi_mask)[0]
        
        # --- 3. SETTINGS ---
        self.num_points = cfg.num_points
        self.overlap_radius = cfg.overlap_radius
        
        
    def __len__(self):
        return len(self.deformed_paths)

    def load_deformed_mesh(self, index):
        path_def = self.deformed_paths[index]
        data_def = np.load(path_def)
        
        tm_def = trimesh.Trimesh(vertices=data_def['vertices'], faces=data_def['faces'])
        dense_def = tm_def.subdivide()
        
        return dense_def.vertices.astype(np.float32)
    
    def __getitem__(self, item):
        verts_und = self.verts_und
        verts_def = self.load_deformed_mesh(item)
        
        # 1. Patch Selection
        seed_idx = np.random.choice(self.valid_roi_indices)
        seed_point = verts_def[seed_idx]
        
        tree = cKDTree(verts_def) 
        _, patch_indices = tree.query(seed_point, k=self.num_points)
        
        # Get Synced Patches
        patch_und = verts_und[patch_indices] 
        patch_def = verts_def[patch_indices] # The raw deformed shape
        
        # 2. Augmentation (FIX: Center -> Rotate -> Translate)
        # If we don't center, the rotation swings the breast far away in 3D space
        centroid_def = np.mean(patch_def, axis=0)
        patch_def_centered = patch_def - centroid_def
        
        T_aug = self.generate_random_transform()
        
        # This is the "Messy" target the network sees
        tgt_aug = tf.transform_points(patch_def_centered, T_aug)
        
        # 3. Ground Truth Pose (FIX: Source -> Target)
        # We need the T that maps patch_und -> tgt_aug
        T_gt = self.generate_transformation_matrix(source=patch_und, target=tgt_aug)
        
        # 4. Overlap (Check Validity)
        # Move source to target using GT to see if they overlap
        src_aligned_to_tgt = tf.transform_points(patch_und, T_gt)
        
        # Since indices are synced, we can check Euclidean distance directly
        # We don't need cKDTree here because patch_und[0] IS patch_def[0]
        residuals = np.linalg.norm(src_aligned_to_tgt - tgt_aug, axis=1)
        overlap_mask = (residuals < self.overlap_radius).astype(np.float32)
        
        # Create correspondences (1:1 match filtered by mask)
        valid_indices = np.where(overlap_mask == 1)[0]
        correspondences = np.stack([valid_indices, valid_indices], axis=1)
        
        data_pair = {
            'src_xyz': torch.from_numpy(patch_und).float(),
            'tgt_xyz': torch.from_numpy(tgt_aug).float(),
            'src_overlap': torch.from_numpy(overlap_mask),
            'tgt_overlap': torch.from_numpy(overlap_mask),
            'correspondences': torch.from_numpy(correspondences).long(),
            'pose': torch.from_numpy(T_gt).float(),
            'idx': item,
        }
        
        if self.transform:
            data_pair = self.transform(data_pair)
        
        return data_pair

    @staticmethod
    def generate_random_transform():
        """Random Rotation + Translation"""
        rot = R_scipy.from_euler('xyz', np.random.uniform(0, 360, 3), degrees=True).as_matrix()
        trans = np.random.uniform(-50, 50, 3) # +/- 50mm translation
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T

    @staticmethod
    def generate_transformation_matrix(source, target): # source, target
        """ calculates the rigid transformation matrix to align patch_points to undeformed_points
        """
        # get centroids
        centroid_src = np.mean(source, axis=0)
        centroid_tgt = np.mean(target, axis=0)
        
        # center the points
        src_centered = source - centroid_src 
        tgt_centered = target - centroid_tgt 
        
        # compute covariance matrix
        # FIX: H = source.T @ target (for Source->Target)
        H = np.dot(src_centered.T, tgt_centered)
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        # source to origin
        T_1 = np.eye(4)
        T_1[0:3, 3] = -centroid_src
        
        # Rotation matrix
        T_2 = np.eye(4)
        T_2[0:3, 0:3] = R
        
        # origin to target
        T_3 = np.eye(4)
        T_3[0:3, 3] = centroid_tgt
        
        return T_3 @ T_2 @ T_1
    