# NetworkX 兼容性补丁应用总结

## 问题描述
在 Python 3.9+ 环境中运行项目时，会遇到以下错误：
```
ImportError: cannot import name 'gcd' from 'fractions'
```

这是因为 `networkx 2.2` 版本仍然尝试从 `fractions` 模块导入 `gcd` 函数，但在 Python 3.5+ 中，`gcd` 函数已经被移动到 `math` 模块。

## 解决方案
在所有导入 `torch_robotics` 或其他可能使用 networkx 的模块之前，添加以下 monkey patch：

```python
import sys
import fractions
import math

# 猴子补丁 (Monkey Patch)：用于兼容性修复
# 修复 networkx 2.2 在 Python 3.9+ 中使用已弃用的 fractions.gcd 的问题
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

from mpd.utils.patches import numpy_monkey_patch

# 应用 numpy 的补丁，解决版本兼容性问题
numpy_monkey_patch()
```

## 已打补丁的文件列表

以下文件已添加 networkx 兼容性补丁：

### 1. 推理脚本
- ✅ `scripts/inference/inference.py` (已有补丁)

### 2. 数据生成脚本
- ✅ `scripts/generate_data/generate_trajectories.py` (新添加)
- ✅ `scripts/generate_data/visualize_trajectories.py` (新添加)
- ✅ `scripts/generate_data/post_process_generated_dataset.py` (新添加)

### 3. 训练脚本
- ✅ `scripts/train/train.py` (新添加)

## 验证
补丁已成功应用，`generate_trajectories.py` 脚本可以正常运行，不再出现 `ImportError`。

## 注意事项
- 这个补丁必须在导入 `torch_robotics` 和其他相关模块**之前**执行
- Flake8 会报告 "module level import not at top of file" 警告，这是预期的，因为我们需要先执行补丁
- 如果有其他脚本遇到相同的错误，可以参照以上模式添加相同的补丁代码

## 日期
2026-01-26
