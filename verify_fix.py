"""
验证训练脚本修复是否已应用
"""
import os
from pathlib import Path

def check_train_py():
    """检查 train.py 是否已修复"""
    train_py = Path('train.py')

    if not train_py.exists():
        print("❌ train.py 不存在")
        return False

    with open(train_py, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查 validate 函数中是否有 autocast
    checks = {
        '混合精度修复（validate中的autocast）': 'if self.use_amp:\n                    with torch.cuda.amp.autocast():',
        'CUDA缓存清理': 'torch.cuda.empty_cache()',
        '新的 AMP API 检查': 'torch.amp.GradScaler' in content or 'torch.cuda.amp.GradScaler' in content
    }

    all_pass = True

    print("=" * 60)
    print("验证修复状态")
    print("=" * 60)

    for check_name, check_pattern in checks.items():
        if isinstance(check_pattern, bool):
            status = "✅" if check_pattern else "❌"
            print(f"{status} {check_name}")
        else:
            if check_pattern in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_pass = False

    print("=" * 60)

    if all_pass:
        print("✅ 所有修复已应用，可以重新运行 python train.py")
    else:
        print("❌ 修复未完全应用，请执行以下步骤：")
        print("  1. 关闭 VSCode")
        print("  2. 删除 __pycache__ 文件夹")
        print("  3. 重新打开 VSCode")
        print("  4. 重新运行 python train.py")

    return all_pass

if __name__ == '__main__':
    check_train_py()
