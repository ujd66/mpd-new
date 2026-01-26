# 兼容性修复说明

## 问题
- `networkx` 2.2 与 Python 3.9+ 不兼容（`fractions.gcd` 已移除）
- `networkx` 2.2 与 NumPy 1.20+ 不兼容（`np.int` 等别名已弃用）

## 已修复脚本
✅ `generate_trajectories.py`  
✅ `launch_generate_trajectories.py`  
✅ `post_process_generated_dataset.py`  
✅ `visualize_trajectories.py`

## 修复方法
所有脚本开头添加了 monkey patch：
```python
import fractions, math
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import numpy as np
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    # ... 等
```

## 其他脚本如遇到类似问题
在文件开头添加相同代码即可。
