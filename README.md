# Harris Corner Detection 课程作业（最终版）

本项目是计算机视觉课程作业，围绕 Harris 角点检测完成了 4 组实验：

1. Harris 算法在不同图像类型上的表现差异
2. 手写 Harris 与 OpenCV Harris 在同一批图像上的检测对比。
3. 参数敏感性实验（`k`、`blockSize`、`sobel_ksize`、`threshold`）。
4. 不同图像角点数量统计与结论分析。

## 1. 项目结构

```text
harris-corner/
├─ data/
│  ├─ raw/                         # 输入图像
│  └─ output/
│     ├─ manual/                   # 手写 Harris 检测图
│     ├─ opencv/                   # OpenCV Harris 检测图
│     ├─ compare/                  # 原图 + 手写 + OpenCV 并排图
│     ├─ sensitivity/              # 参数敏感性结果（按图片分目录）
│     └─ stats/
│        └─ corner_counts.csv      # 角点数量统计结果
├─ src/
│  ├─ harris/
│  │  ├─ harris.py                 # 手写 Harris 响应
│  │  └─ nms.py                    # 非极大值抑制
│  ├─ experiment_compare.py        # 实验1：批量检测对比
│  ├─ experiment_param_sensitivity.py # 实验2：参数敏感性
│  ├─ stat_corner_counts.py        # 实验3：角点数量统计
│  ├─ run_manual_harris.py.py      # 早期脚本（保留）
│  └─ run_opencv_harris.py         # 早期脚本（保留）
├─ reports/
│  └─ harris.md
├─ pyproject.toml
└─ README.md
```

## 2. 环境与依赖

- Python `>= 3.12`
- 主要依赖：`opencv-python`、`numpy`、`matplotlib`、`tqdm`、`scikit-image`



## 3. 快速复现实验

### 3.1 实验1：手写 vs OpenCV 对比

```bash
python src/experiment_compare.py
```

输出：

- `data/output/manual/`
- `data/output/opencv/`
- `data/output/compare/`

### 3.2 实验2：参数敏感性（默认对全部图片）

```bash
python src/experiment_param_sensitivity.py
```

默认参数范围：

- `k`: `0.02,0.04,0.06,0.08`
- `blockSize`: `2,3,5,7`
- `sobel_ksize`: `3,5,7`
- `threshold`: `0.005,0.01,0.02,0.05`

可选：仅跑单张图

```bash
python src/experiment_param_sensitivity.py --image data/raw/building.jpg
```

输出：

- `data/output/sensitivity/<image_name>/k_sweep.jpg`
- `data/output/sensitivity/<image_name>/block_size_sweep.jpg`
- `data/output/sensitivity/<image_name>/sobel_ksize_sweep.jpg`
- `data/output/sensitivity/<image_name>/threshold_sweep.jpg`

### 3.3 实验3：角点数量统计

```bash
python src/stat_corner_counts.py
```

输出：

- `data/output/stats/corner_counts.csv`

统计字段：

- `manual_nms`：手写 Harris + NMS
- `opencv_threshold_only`：OpenCV Harris + 阈值筛选
- `opencv_nms`：OpenCV Harris + NMS

## 4. 方法说明

### 4.1 手写 Harris

`src/harris/harris.py` 实现：

1. Sobel 计算梯度 `Ix`、`Iy`
2. 构造二阶矩项 `Ix^2`、`Iy^2`、`IxIy`
3. 高斯滤波进行窗口加权
4. 计算响应：`R = det(M) - k * trace(M)^2`

### 4.2 NMS

`src/harris/nms.py` 实现：

1. 阈值筛选 `R > threshold_ratio * max(R)`
2. 膨胀求局部极大值
3. 取交集作为最终角点

### 4.3 OpenCV Harris

调用：

```python
cv2.cornerHarris(gray, blockSize, ksize, k)
```

在实验中分别测试了“仅阈值筛选”和“阈值 + NMS”两种后处理。

## 5. 报告文件

- `reports/harris.md`


## 6. 备注

- `run_manual_harris.py.py` 与 `run_opencv_harris.py` 为早期实验脚本，当前推荐使用 `experiment_*.py` 与 `stat_corner_counts.py`。
- `main.py` 当前未作为入口脚本使用。
