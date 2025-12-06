"""
图像处理模块
处理Camera和Sonar图像的加载、裁剪、resize和归一化
"""
import numpy as np
from PIL import Image
from pathlib import Path


class ImageProcessor:
    """图像处理器"""

    def __init__(self, config):
        """
        Args:
            config: 配置字典，包含image部分的配置
        """
        self.camera_config = config['image']['camera']
        self.sonar_config = config['image']['sonar']

    def process_camera(self, img_path):
        """
        处理Camera图像

        Args:
            img_path: 图像路径

        Returns:
            np.ndarray: [3, H, W] 归一化后的图像
        """
        # 使用PIL读取（支持中文路径）
        img = Image.open(str(img_path))

        # Resize
        target_size = tuple(self.camera_config['target_size'])
        img = img.resize(target_size, Image.BILINEAR)

        # 转换为numpy数组 [H, W, 3]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 归一化
        mean = np.array(self.camera_config['normalize_mean'], dtype=np.float32)
        std = np.array(self.camera_config['normalize_std'], dtype=np.float32)
        img_array = (img_array - mean) / std

        # 转换为 [C, H, W]
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def process_sonar(self, img_path):
        """
        处理Sonar图像（ROI裁剪 + resize + 归一化）

        Args:
            img_path: 图像路径

        Returns:
            np.ndarray: [3, H, W] 归一化后的图像
        """
        # 使用PIL读取
        img = Image.open(str(img_path))

        # ROI裁剪
        if self.sonar_config['use_roi']:
            y1, y2, x1, x2 = self.sonar_config['roi_coords']
            img = img.crop((x1, y1, x2, y2))

        # Resize
        target_size = tuple(self.sonar_config['target_size'])
        img = img.resize(target_size, Image.BILINEAR)

        # 转换为numpy数组
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 如果是灰度图，转换为3通道
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        # 归一化
        mean = np.array(self.sonar_config['normalize_mean'], dtype=np.float32)
        std = np.array(self.sonar_config['normalize_std'], dtype=np.float32)
        img_array = (img_array - mean) / std

        # 转换为 [C, H, W]
        img_array = np.transpose(img_array, (2, 0, 1))

        return img_array

    def batch_process(self, camera_paths, sonar_paths):
        """
        批量处理图像

        Args:
            camera_paths: Camera图像路径列表
            sonar_paths: Sonar图像路径列表

        Returns:
            tuple: (camera_batch, sonar_batch)
                camera_batch: [T, 3, H, W]
                sonar_batch: [T, 3, H, W]
        """
        camera_batch = []
        sonar_batch = []

        for cam_path, son_path in zip(camera_paths, sonar_paths):
            cam_img = self.process_camera(cam_path)
            son_img = self.process_sonar(son_path)
            camera_batch.append(cam_img)
            sonar_batch.append(son_img)

        camera_batch = np.stack(camera_batch, axis=0)  # [T, 3, H, W]
        sonar_batch = np.stack(sonar_batch, axis=0)

        return camera_batch, sonar_batch
