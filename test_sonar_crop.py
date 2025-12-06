"""
测试Sonar图像裁剪效果
自动检测有效ROI区域并展示裁剪结果
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def detect_sonar_roi(img_path, threshold=3):
    """
    自动检测sonar图像的有效ROI区域
    使用更低的阈值和更大的padding来捕获完整的扇形区域（包括暗淡部分）

    Args:
        img_path: 图像路径
        threshold: 二值化阈值（降低以捕获暗区）

    Returns:
        (y1, y2, x1, x2): ROI坐标
    """
    # 使用PIL读取图像（支持中文路径）
    pil_img = Image.open(str(img_path))
    img = np.array(pil_img)

    # 如果是RGB，转换为BGR供OpenCV使用
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用更低的阈值来检测包括暗区的完整扇形区域
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 找到所有轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"警告: 未检测到有效区域，使用全图")
        return (0, img.shape[0], 0, img.shape[1])

    # 找到最大轮廓的边界框
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 添加更大的padding（30%），确保捕获完整扇形
    padding_x = int(w * 0.35)
    padding_y = int(h * 0.35)

    y1 = max(0, y - padding_y)
    y2 = min(img.shape[0], y + h + padding_y)
    x1 = max(0, x - padding_x)
    x2 = min(img.shape[1], x + w + padding_x)

    return (y1, y2, x1, x2)


def crop_and_resize(img, roi, target_size=(320, 320)):
    """
    根据ROI裁剪并resize图像

    Args:
        img: 原始图像
        roi: (y1, y2, x1, x2)
        target_size: 目标尺寸

    Returns:
        处理后的图像
    """
    y1, y2, x1, x2 = roi
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def visualize_crop_results(sonar_dir, sample_indices, output_path=None):
    """
    可视化多个sonar图像的裁剪效果

    Args:
        sonar_dir: sonar图像目录
        sample_indices: 要展示的样本索引列表（如[0, 10, 50, 100]）
        output_path: 保存路径（可选）
    """
    sonar_dir = Path(sonar_dir)
    n_samples = len(sample_indices)

    fig, axes = plt.subplots(n_samples, 3, figsize=(15, n_samples * 4))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(sample_indices):
        # 构建文件路径（索引从1开始，00001.png）
        img_filename = f"{sample_idx + 1:05d}.png"
        img_path = sonar_dir / img_filename

        if not img_path.exists():
            print(f"警告: 文件不存在 {img_path}")
            continue

        # 使用PIL读取原始图像（支持中文路径）
        pil_img = Image.open(str(img_path))
        img = np.array(pil_img)

        # 转换为RGB（如果需要）
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img_rgb = img  # PIL已经是RGB
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测ROI
        roi = detect_sonar_roi(img_path)
        y1, y2, x1, x2 = roi

        # 裁剪并resize（使用PIL操作）
        pil_img_cropped = pil_img.crop((x1, y1, x2, y2))
        pil_img_resized = pil_img_cropped.resize((320, 320), Image.BILINEAR)
        cropped_rgb = np.array(pil_img_resized)

        # 在原图上绘制ROI框
        img_with_box = img_rgb.copy()
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 显示
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f'原图 (样本{sample_idx}: {img_filename})\n尺寸: {img_rgb.shape[:2]}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(img_with_box)
        axes[idx, 1].set_title(f'ROI标注\nROI: [{y1}:{y2}, {x1}:{x2}]\n尺寸: {y2-y1}x{x2-x1}')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cropped_rgb)
        axes[idx, 2].set_title(f'裁剪并Resize\n目标尺寸: 320x320')
        axes[idx, 2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存至: {output_path}")
    else:
        plt.show()

    plt.close()

    # 返回统计信息
    print("\n=== ROI统计信息 ===")
    all_rois = []
    for i in range(min(100, 3828)):  # 检查前100张图像
        img_filename = f"{i + 1:05d}.png"
        img_path = sonar_dir / img_filename
        if img_path.exists():
            roi = detect_sonar_roi(img_path)
            all_rois.append(roi)

    if all_rois:
        y1_vals = [r[0] for r in all_rois]
        y2_vals = [r[1] for r in all_rois]
        x1_vals = [r[2] for r in all_rois]
        x2_vals = [r[3] for r in all_rois]

        print(f"检查了前{len(all_rois)}张图像:")
        print(f"  Y范围: [{int(np.mean(y1_vals))}:{int(np.mean(y2_vals))}] (平均)")
        print(f"  X范围: [{int(np.mean(x1_vals))}:{int(np.mean(x2_vals))}] (平均)")
        print(f"  高度: {int(np.mean([y2-y1 for y2, y1 in zip(y2_vals, y1_vals)]))} (平均)")
        print(f"  宽度: {int(np.mean([x2-x1 for x2, x1 in zip(x2_vals, x1_vals)]))} (平均)")

        # 建议固定ROI
        avg_y1, avg_y2 = int(np.mean(y1_vals)), int(np.mean(y2_vals))
        avg_x1, avg_x2 = int(np.mean(x1_vals)), int(np.mean(x2_vals))
        print(f"\n建议的固定ROI坐标: [{avg_y1}:{avg_y2}, {avg_x1}:{avg_x2}]")


if __name__ == "__main__":
    # 配置路径
    SONAR_DIR = "D:/水下小目标项目/数据集-收集中/新数据/Sunboat_03-09-2023/2023-09-03-07-58-37/sonar"
    OUTPUT_PATH = "D:/Vscodeprogram/underwater_multimodal_prediction/results/figures/sonar_crop_test.png"

    # 选择要展示的样本（索引0, 10, 100, 500, 1000）
    sample_indices = [0, 10, 100, 500, 1000]

    print("开始测试Sonar图像裁剪...")
    print(f"Sonar目录: {SONAR_DIR}")
    print(f"测试样本索引: {sample_indices}")
    print()

    # 创建输出目录
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 运行可视化
    visualize_crop_results(SONAR_DIR, sample_indices, OUTPUT_PATH)

    print("\n测试完成！")
    print(f"请查看: {OUTPUT_PATH}")
