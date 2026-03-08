---
title: "Harris"
date: 2026-03-06
categories: ["cv"]
draft: false 
---

> 课程作业：
>
> Harris特征点检测器-兴趣点检测
>
> **理论部分**
>
> - 掌握Harris特征点检测的原理
>
> **练习部分**
>
> - 使用OpenCV集成的Harris特征点检测器实现图像兴趣点检测

# Harris特征点检测器-兴趣点检测 



## 1. 相关理论 & 概念

### 1.1 图像的表示

计算机中的图像是一个二维函数：
$$
I(x,y)
$$
含义：

- $x,y$ ： 像素坐标
- $I(x,y)$ : 像素灰度值 [ 亮度信息 $I(x,y) \in (0,255)$ ]
- 准确来说 $I(x,y)$ 应该表示该位置的像素强度，比如RGB的三通道分别对应颜色的强度，但Harris 通常只关心亮度，所以这里使用灰度图

对应代码：

```python
import cv2
img = cv2.imread("image.jpg",0) # 加0是读入灰度图，不加0是读入RGB
# 此时img是一个二维numpy矩阵
# 例如：
# [[120 122 125]
#  [130 135 140]
#  [115 118 123]]
```



### 1.2 图像梯度

#### 1.2.1 图像梯度(gradient)：

$$
∇I=
\begin{bmatrix}
I_x=\frac{∂I}{∂x} \\
I_y=\frac{∂I}{∂y}
\end{bmatrix}
$$

含义：

- $I_x$ : 水平方向变化
- $I_y$ : 垂直方向变化

梯度大小：
$$
|∇I| = \sqrt{I_x^2+I_y^2}
$$
梯度方向：
$$
θ=arctan{\frac{I_y}{I_x}}
$$
梯度可以理解为：**像素亮度变化的速度和方向** , 变化越快 → 梯度越大。



#### 1.2.2 连续函数的导数：

在连续函数中，导数的定义为：
$$
f'(x)=\lim_{h \to 0}\frac{f(x+h)-f(x)}{h}
$$

- 变化率 = 很小距离内函数的变化
- 但图像的像素值是离散的，h不能无限小，无法使用连续函数的导数定义求梯度

#### 1.2.3 离散函数的导数：有限差分

在数值计算中，导数通常使用差分近似

中心差分：
$$
f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}
$$

- $h$: 采样间隔
- 在数字图像中，相邻像素之间的距离为，所以 $h=1$

$$
f'(x) \approx \frac{f(x+1)-f(x-1)}{2}
$$

差分可以写成卷积形式

中心差分可以写成：
$$
\begin{bmatrix}
-1 & 0 & 1
\end{bmatrix}
$$

推导过程
$$
f'(x) \approx \frac{f(x+1)-f(x-1)}{2}\\
改写\\
f'(x) \approx -\frac{1}{2}f(x-1) + 0f(x) + \frac{1}{2}f(x+1)
$$
注意这里的结构：

| 位置  | 权重           |
| ----- | -------------- |
| $x-1$ | $-\frac12$     |
| $x$   | $0$            |
| $x+1$ | $+\frac{1}{2}$ |

这其实就是一个 **局部加权和**。

离散卷积定义：
$$
g(x)=\sum_{k=-\infty}^{\infty} f(x+k)\,h(k)
$$

- $f$: 源信号
- $h$: 卷积核

如果卷积核只有三个元素：
$$
h =
\begin{bmatrix}
h(-1) , h(0) , h(1)
\end{bmatrix}
$$
那么卷积就是：
$$
g(x)=h(−1)f(x−1)+h(0)f(x)+h(1)f(x+1)
$$

- **卷积核只是把差分公式写成“局部加权求和”的形式**。
- 中心差分本质上就是对 **邻域像素做线性组合**，而卷积正是做这件事。
- sobel算子就是在用这个中心差分的卷积形式，卷积出的结果就是导数的离散近似

#### 1.2.4 Sobel 算子

x方向：
$$
S_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$
y方向：
$$
S_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$
可以看到两个卷积核就是换了方向，这里以x方向为例讲解：

根据中心差分及其卷积形式，得到了一个一维导数算子，但图像是二维函数，如果直接在每一行用这个一维核，会对噪声非常敏感

解决方法是：在另一个方向上做平滑

计算x方向导数的时候，我们希望：

- x方向：计算梯度
- y方向：做平均（降噪）

这就是Sobel的设计思路

最简单的平均是：
$$
\begin{bmatrix}
1 & 1 & 1
\end{bmatrix}
$$
但Sobel使用的是：
$$
\begin{bmatrix}
1 & 2 & 1
\end{bmatrix}
$$
不使用 $[111]$ 的原因：高斯平滑思想 ； 图像处理中通常使用 高斯滤波降噪，这里没有对高斯滤波深挖，它归一化后的特点是：

- 中心权重大
- 两边权重小

使用 $[121]$ 更接近高斯近似

Sobel 实际上是两个核的外积

Sobel 在做三件事：

-  **左右像素相减** → 求变化率
-  **上下做加权平均** → 降噪
-  **中心权重更大** → 更接近高斯滤波

对应OpenCV代码：

```python
Ix = cv2.Sobel(img,cv2.CV_64F,1,0)
Iy = cv2.Sobel(img,cv2.CV_64F,0,1)
```



### 1.3 角点(Corner)

梯度在图像中的意义

观察一条灰度变化：

- ```txt
  10 10 10 10 10
  ```

  - 变化为0
  - 这是**平坦区域**

- ```txt
  10 10 10 200 200
  ```

  - 变化很大
  - 这是**边缘**

- ```txt
  10 10 10 
  10 200 200
  10 200 200
  ```

  - 在两个方向都变化很大
  - 这是**角点** （在两个方向上都有明显灰度变化的点）

Harris 判断角点的基础：

| 区域 | 梯度         |
| ---- | ------------ |
| 平坦 | ≈0           |
| 边缘 | 单方向大     |
| 角点 | 两个方向都大 |

### 1.4 Harris矩阵

描述一个像素邻域内的梯度分布情况，从而判断该点是平坦区、边缘还是角点。

Harris 的核心思想就是：观察一个小窗口移动时，图像灰度变化有多大。

核心数学对象：
$$
M =
\begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}
$$
判断角点：

关键在于特征值

设$ λ_1,λ_2$ 是矩阵M的特征值

这两个值代表： 窗口在两个主方向上的变化程度

| 区域 | 特征值                          |
| ---- | ------------------------------- |
| 平坦 | $λ_1 \approx 0 ,λ_2 \approx 0 $ |
| 边缘 | $ λ_1 \gg λ_2$                  |
| 角点 | $λ_1,λ_2 \gg 0$                 |



推导过程：

1. 定义窗口平移后的变化量

   设图像为 $I(x,y)$

   窗口移动$(u,v)$后的变化量：
   $$
   E(u,v)=\sum_{x,y} w(x,y)[I(x+u,y+v)-I(x,y)]^2
   $$

   - $w(x,y)$: 窗口权重（通常是高斯）
   - $(u,v)$: 窗口移动
   - 如果移动后差值大 → 说明这个位置变化剧烈

2. 用泰勒展开近似

   使用一阶泰勒展开：
   $$
   I(x+u,y+v) \approx I(x,y)+I_xu+I_yv
   $$
   其中：
   $$
   I_x=\frac{∂I}{∂x} \: I_y=\frac{∂I}{∂y}
   $$
   代入 $E(u,v)$ ：
   $$
   E(u,v) \approx \sum_{x,y} w(x,y)[I_xu+I_yv]^2
   $$
   展开平方：
   $$
   [I_xu+I_xv]^2 = I_x^2u^2 + 2I_xI_yuv + I_y^2v^2
   $$
   代入 $E(u,v)$ ：
   $$
   E(u,v) \approx \sum_{x,y} w(x,y)(I_x^2u^2 + 2I_xI_yuv + I_y^2v^2)
   $$
   提出 $u,v$ ：

   - $u,v$ 是窗口位移 (常数)
   - 求和只对像素做

   因此可以整理为：
   $$
   E(u,v) \approx u^2\sum_{x,y} w(x,y)I_x^2 +  2uv \sum_{x,y} w(x,y)I_xI_y + v^2\sum_{x,y} w(x,y)I_y^2
   $$
   定义三个量：
   $$
   A =\sum wI_x^2 \\
   B =\sum wI_xI_y \\
   C =\sum wI_y^2
   $$
   于是：
   $$
   E(u,v) \approx Au^2 + 2Buv + Cv^2
   $$
   这是一个二次型表达式，可以写成矩阵形式
   $$
   \begin{bmatrix}
   u & v
   \end{bmatrix}
   \begin{bmatrix}
   A & B \\
   B & C
   \end{bmatrix}
   \begin{bmatrix}
   u \\
   v
   \end{bmatrix}
   $$
   
3. 提取矩阵形式

   整理后：
   $$
   E(u,v) =
   \begin{bmatrix}
   u & v
   \end{bmatrix}
   M
   \begin{bmatrix}
   u \\
   v
   \end{bmatrix}
   $$
   其中：
   $$
   M = \sum w(x,y)
   \begin{bmatrix}
   I_x^2 & I_x I_y \\
   I_x I_y & I_y^2
   \end{bmatrix}
   $$
   
4. 结构张量的含义

   | 元素     | 含义           |
   | -------- | -------------- |
   | $I_x^2$  | x方向梯度强度  |
   | $I_y^2$  | y方向梯度强度  |
   | $I_xI_y$ | 梯度方向相关性 |


### 1.5 Harris响应函数

在早期计算资源紧张的时期，求特征值要开方，复杂度高；并且Harris需要判断的是两个特征值的大小，不需要精确的特征值，所以Harris提出一个响应函数：
$$
R = \det(M) - k(\operatorname{trace}(M))^2
$$

- $det(M) =  λ_1 λ_2 = AC-B^2$
- $trace(M) =  λ_1 + λ_2 = A+C$
- 参数 $k \approx 0.04 \sim 0.06$

响应函数更稳定并且能自然区分边缘

| R值   | 含义 |
| ----- | ---- |
| R < 0 | 边缘 |
| R ≈ 0 | 平坦 |
| R > 0 | 角点 |

### 1.6 非极大值抑制

> （Non-Maximum Suppression, NMS）

Harris 响应函数计算完成后，每个像素都会得到一个响应值 $R$。

但一个真实的角点通常会在 **邻域内产生一片较大的响应值区域**，如果直接取所有 $R>0$ 的点，会得到很多重复点。

因此需要 **非极大值抑制（NMS）**：
 只保留 **局部邻域内响应值最大的像素**，其余全部抑制掉。

步骤：

1. 设定一个窗口（例如 $3\times3$ 或 $5\times5$）

2. 对每个像素 $(x,y)$：

   - 在邻域窗口内比较响应值 $R$

3. 如果：
   $$
   R(x,y) = \max_{\Omega}R
   $$
   

   则保留该点，否则抑制为0

其中：

- $\Omega$ 表示邻域窗口
- $ \max_{\Omega}R$ 表示窗口内最大响应值

通常还会结合 **阈值过滤**：
$$
R(x,y)>T
$$

- $T$ 为经验阈值
- 用于去除弱响应

最终保留下来的点同时满足：
$$
\begin{cases} 
R(x,y) > T \\ R(x,y) = \max_{\Omega}R 
\end{cases}
$$

## 2. 方法实现

本实验使用 **OpenCV + Python** 实现 Harris 特征点检测，并与 OpenCV 自带函数进行对比验证。

实现目标：

1. **手动实现 Harris 算法**
2. **使用 OpenCV 自带 Harris**
3. 对结果进行 **可视化对比**

实验环境：

| 项目           | 配置           |
| -------------- | -------------- |
| 操作系统       | Arch Linux     |
| Python版本     | Python 3.12.12 |
| Python环境管理 | mise           |
| Python包管理   | uv             |
| 图像处理库     | OpenCV         |
| 数值计算库     | NumPy          |
| 可视化库       | Matplotlib     |

### 2.1 数据输入

实验使用普通自然图像作为测试数据，例如：

- 建筑物
- 书本
- 桌角
- 棋盘格

这些图像具有 **明显角点结构**。

读取图像：

```python
import cv2
import numpy as np

# 读取图像并转灰度：
img = cv2.imread("image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris 需要 float 类型
gray = np.float32(gray)
```

说明：

- Harris 使用 **灰度图**
- OpenCV Harris 要求 **float32**

### 2.2 Harris算法实现

按照 Harris 理论流程实现：

算法步骤：

```txt
图像
 ↓
Sobel 梯度
 ↓
梯度乘积
 ↓
高斯窗口求和
 ↓
Harris响应
 ↓
阈值筛选
 ↓
非极大值抑制
 ↓
角点输出
```

#### 2.2.1  Sobel 梯度计算

$$
I_x=\frac{∂I}{∂x} \\ I_y=\frac{∂I}{∂y}
$$

代码：

```python
# x方向梯度
Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
# y方向梯度
Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
```

说明：

| 参数    | 含义            |
| ------- | --------------- |
| 1,0     | x方向求导       |
| 0,1     | y方向求导       |
| ksize=3 | 使用3×3 Sobel核 |

#### 2.2.2 梯度乘积

$$
I_x^2 ,\: I_y^2\:, I_x I_y
$$

代码：

```python
Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy
```



#### 2.2.3 高斯加权求和（窗口积分）

$$
A = \sum w(x,y) I_x^2\\
B = \sum w(x,y) I_x I_y\\
C = \sum w(x,y) I_y^2\\
$$

$w(x,y)$ 为高斯权重

得到 Harris 矩阵:
$$
M =
\begin{bmatrix}
A & B \\
B & C
\end{bmatrix}
$$

代码：

```python
# 使用高斯滤波实现窗口加权求和
A = cv2.GaussianBlur(Ix2, (3,3), 1)
B = cv2.GaussianBlur(Ixy, (3,3), 1)
C = cv2.GaussianBlur(Iy2, (3,3), 1)
```

说明：

- 高斯滤波相当于 **加权窗口积分**
- 可以减少噪声影响

#### 2.2.4 Harris 响应函数

$$
R = \det(M) - k(\operatorname{trace}(M))^2 \\
\det(M) = AC - B^2 \\
\operatorname{trace}(M) = A + C\\
k \approx 0.04 \sim 0.06
$$

代码：

```python
k = 0.04
detM = A * C - B * B
traceM = A + C
R = detM - k * (traceM ** 2)
```



#### 2.2.5 阈值筛选

为了去除弱响应点，需要设定阈值：
$$
R(x,y) > T
$$

- $T$ 为经验阈值

代码：

```python
threshold = 0.01 * R.max()
corner_mask = R > threshold
```

说明：

经验阈值通常为：

```txt
0.01 ~ 0.1 × max(R)
```



#### 2.2.6 非极大值抑制

角点通常形成一小片区域，需要保留 **局部最大值**。

数学表达：
$$
R(x,y) = \max_{\Omega} R
$$
实现方法：

利用膨胀操作：

```python
R_dilate = cv2.dilate(R, None)
local_max = (R == R_dilate)
corner = corner_mask & local_max
```

解释：

| 操作          | 作用               |
| ------------- | ------------------ |
| dilate        | 计算邻域最大值     |
| R == R_dilate | 判断是否为局部最大 |

#### 2.2.7 角点可视化

最终得到角点集合：
$$
\{(x_i, y_i)\}
$$

代码：

```python
ys, xs = np.where(corner)
points = list(zip(xs, ys))

# 可视化角点
result = img.copy()

for x, y in points:
    cv2.circle(result, (x, y), 3, (0,0,255), 1)

cv2.imshow("corners", result)
cv2.waitKey(0)
```

> ```python
> # 完整版代码
> import cv2
> import numpy as np
> 
> img = cv2.imread("image.png")
> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
> gray = np.float32(gray)
> 
> # 1 Sobel梯度
> Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
> Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
> 
> # 2 梯度乘积
> Ix2 = Ix * Ix
> Iy2 = Iy * Iy
> Ixy = Ix * Iy
> 
> # 3 高斯窗口求和
> A = cv2.GaussianBlur(Ix2, (3,3), 1)
> B = cv2.GaussianBlur(Ixy, (3,3), 1)
> C = cv2.GaussianBlur(Iy2, (3,3), 1)
> 
> # 4 Harris响应
> k = 0.04
> detM = A * C - B * B
> traceM = A + C
> R = detM - k * traceM**2
> 
> # 5 阈值筛选
> threshold = 0.01 * R.max()
> corner_mask = R > threshold
> 
> # 6 非极大值抑制
> R_dilate = cv2.dilate(R, None)
> local_max = R == R_dilate
> corner = corner_mask & local_max
> 
> # 7 输出角点
> ys, xs = np.where(corner)
> 
> result = img.copy()
> for x, y in zip(xs, ys):
>     cv2.circle(result, (x,y), 3, (0,0,255), 1)
> 
> cv2.imshow("corners", result)
> cv2.waitKey(0)
> ```
>
> OpenCV 自带 Harris
>
> 实际上 OpenCV 已经封装好了
>
> ```python
> dst = cv2.cornerHarris(gray, 2, 3, 0.04)
> # 2  → blockSize (窗口大小)
> # 3  → Sobel核
> # 0.04 → k
> ```
>

## 3. 实验设计

本实验的目标是验证 **Harris 特征点检测算法** 在实际图像中的表现，并分析其对不同图像结构和参数设置的敏感性。

实验主要包含以下几个部分：

1. **基础角点检测实验**
2. **OpenCV 实现对比**
3. **角点数量统计分析**
4. **参数敏感性实验**

实验流程如下：

```txt
输入图像
 ↓
灰度化
 ↓
Harris角点检测
 ↓
阈值筛选
 ↓
非极大值抑制
 ↓
角点可视化
 ↓
统计分析
```

实验使用多种类型图像：

| 图像类型 | 示例         | 特点         |
| -------- | ------------ | ------------ |
| 建筑物   | 楼房         | 大量直线交点 |
| 棋盘格   | checkerboard | 密集规则角点 |
| 室内场景 | 桌子/书本    | 少量角点     |
| 自然场景 | 动物         | 纹理复杂     |

选择这些图像的原因：

- Harris 主要用于 **检测几何结构明显的角点**
- 不同图像结构可以体现算法的优缺点

## 4. 实验结果与分析

### 4.1 基础检测

#### 4.1.1 实验目的

验证 Harris 角点检测在不同图像类型上的表现差异，选取以下四类图像：

- 棋盘格：`chessboard.png`
- 建筑物：`building.jpg`
- 室内场景：`box_in_scene.png`
- 动物：`baboon.jpg`

#### 4.1.2 实验设置

- 算法：手写 Harris（Sobel + 二阶矩阵 + 响应函数 + NMS）
- 参数：`k=0.04`，`threshold_ratio=0.01`
- 输入目录：`data/raw/`
- 输出目录：`data/output/manual/`

#### 4.1.3 检测结果（手写 Harris）

- 棋盘格：![chessboard.png](https://pic.zylona.site/i/2026/03/08/69acfc6975f4d.png)

- 建筑物：![building.jpg](https://pic.zylona.site/i/2026/03/08/69acfc6ac571a.jpg)

- 室内场景：

  ![box_in_scene.png](https://pic.zylona.site/i/2026/03/08/69acfc699f8d6.png)

- 动物：

  ![baboon.jpg](https://pic.zylona.site/i/2026/03/08/69acfc6ac511b.jpg)

角点数量统计（同一参数）：

| 图像 | 角点数量 |
|---|---:|
| chessboard.png | 303 |
| building.jpg | 1553 |
| box_in_scene.png | 799 |
| baboon.jpg | 3060 |

#### 4.1.4 结果分析

1. 棋盘格：
规则交叉结构明显，理论上角点位置稳定。由于 NMS 和阈值筛选，最终角点数量适中，主要集中在网格交点附近。

2. 建筑物：
窗框、边缘交汇和重复结构较多，因此角点数量明显高于棋盘格，且在纹理密集区域更集中。

3. 室内场景：
目标物体与背景共同产生角点，数量中等。受透视、噪声和弱纹理区域影响，角点分布不均匀。

4. 动物图像：
毛发等高频纹理会产生大量局部强响应，导致角点数量最高。这说明 Harris 对纹理细节非常敏感，容易在“纹理点”上产生响应。

#### 4.1.5 结论

- Harris 对结构化角点（如棋盘交点）检测稳定。
- 在自然纹理场景（如动物毛发）中会产生大量角点，需结合阈值与 NMS 控制密度。
- 图像内容复杂度与纹理强度是角点数量差异的主要原因。



### 4.2 OpenCV 对比实验

#### 4.2.1 实验目的

对比手写实现与 OpenCV 封装在同一批图像上的检测效果差异，并分析原因。

#### 4.2.2 实验设置

- 输入图像：`data/raw/` 全部图像
- 手写 Harris 参数：`k=0.04`，`threshold_ratio=0.01`，含 NMS
- OpenCV Harris 参数：`blockSize=2`，`ksize=3`，`k=0.04`，`threshold=0.01*max(R)`
- 输出目录：
  - 手写：`data/output/manual/`
  - OpenCV：`data/output/opencv/`
  - 并排对比：`data/output/compare/`

#### 4.2.3 可视化对比

- `chessboard.png`：![chessboard.png](https://pic.zylona.site/i/2026/03/08/69acfd3644b92.png)
- `building.jpg`：![building.jpg](https://pic.zylona.site/i/2026/03/08/69acfd34eb920.jpg)
- `box_in_scene.png`：![box_in_scene.png](https://pic.zylona.site/i/2026/03/08/69acfd3595e71.png)
- `baboon.jpg`：![baboon.jpg](https://pic.zylona.site/i/2026/03/08/69acfd3630f84.jpg)

#### 4.2.4 定量结果

按当前脚本实现（OpenCV: 阈值筛选，不含 NMS）

| 图像 | 手写 Harris | OpenCV Harris |
|---|---:|---:|
| baboon.jpg | 3060 | 48464 |
| box.png | 529 | 8959 |
| box_in_scene.png | 799 | 11721 |
| building.jpg | 1553 | 22991 |
| chessboard.png | 303 | 4525 |
| graf1.png | 852 | 13420 |

公平对齐后（OpenCV 响应也加同样 NMS）

| 图像 | 手写 Harris | OpenCV + NMS |
|---|---:|---:|
| baboon.jpg | 3060 | 4083 |
| box.png | 529 | 619 |
| box_in_scene.png | 799 | 826 |
| building.jpg | 1553 | 1636 |
| chessboard.png | 303 | 336 |
| graf1.png | 852 | 975 |

#### 4.2.5 分析

1. 当前实验中 OpenCV 结果显著更多，核心原因不是算法本体，而是后处理差异：
手写实现使用 NMS 保留局部极大值；OpenCV 实现仅阈值筛选，保留了大量相邻高响应像素。

2. 在统一后处理（都做 NMS）后，两者角点数量接近，说明两种实现在角点位置趋势上是一致的。

3. OpenCV 封装的优势在于工程稳定性和速度，手写实现的优势在于可解释性和参数可控性。

### 4.3 角点数量统计分析

#### 4.3.1 统计脚本

脚本：`src/stat_corner_counts.py`

功能：

- 读取 `data/raw/` 中所有图像
- 统计每张图在三种设置下的角点数量
  - `manual_nms`：手写 Harris + NMS
  - `opencv_threshold_only`：OpenCV Harris + 阈值筛选
  - `opencv_nms`：OpenCV Harris + NMS
- 输出 CSV：`data/output/stats/corner_counts.csv`

#### 4.3.2 统计结果

CSV 原始结果：

| image | manual_nms | opencv_threshold_only | opencv_nms |
|---|---:|---:|---:|
| baboon.jpg | 3060 | 48464 | 4083 |
| box.png | 529 | 8959 | 619 |
| box_in_scene.png | 799 | 11721 | 826 |
| building.jpg | 1553 | 22991 | 1636 |
| chessboard.png | 303 | 4525 | 336 |
| graf1.png | 852 | 13420 | 975 |

汇总统计：

- `manual_nms`：总计 `7096`，均值 `1182.67`，最小 `303`（`chessboard.png`），最大 `3060`（`baboon.jpg`）
- `opencv_threshold_only`：总计 `110080`，均值 `18346.67`
- `opencv_nms`：总计 `8475`，均值 `1412.50`

比例观察：

- `opencv_threshold_only / manual_nms` 约在 `14.67 ~ 16.94` 倍
- `opencv_nms / manual_nms` 约在 `1.03 ~ 1.33` 倍

#### 4.3.3 概括性结论

1. 不同图像角点数量差异明显：
`baboon.jpg` 最高，`chessboard.png` 最低，说明纹理复杂度对 Harris 响应影响很大。

2. 仅阈值筛选会显著放大角点数量：
OpenCV 在“不加 NMS”时角点数量约为手写结果的 15 倍，主要保留了大量相邻响应像素。

3. 统一 NMS 后，两种实现结果接近：
OpenCV + NMS 与手写 Harris + NMS 的数量接近，说明算法本体趋势一致，差异主要来自后处理策略。

#### 4.3.4 分析

1. 图像内容因素：
自然纹理（如动物毛发）含大量高频细节，局部梯度变化剧烈，导致角点响应密集；规则场景（棋盘格）结构集中，角点更稀疏且可控。

2. 后处理因素：
阈值只做“强度过滤”，不会去重邻域内重复响应；NMS 会保留局部极大值并抑制邻域冗余点，因此角点数量显著下降。

3. 可比性因素：
    跨方法对比时必须统一后处理流程（尤其是否使用 NMS），否则统计结果会被流程差异主导，掩盖算法本身差异。

  


### 4.4 参数敏感性实验

#### 4.4.1 实验目的

分析 Harris 检测对关键参数的敏感性，观察参数变化如何影响角点数量与分布。

#### 4.4.2 实验设置

- 算法：OpenCV `cv2.cornerHarris`
- 实验方式：单变量实验（一次只改一个参数，其余固定）
- 数据：对 `data/raw/` 每张图像分别执行
- 输出目录：`data/output/sensitivity/<image_name>/`

固定参数（默认）：

- `k=0.04`
- `blockSize=2`
- `sobel_ksize=3`
- `threshold=0.01`

#### 4.4.3 参数测试范围

- `k`：`[0.02, 0.04, 0.06, 0.08]`
- `blockSize`：`[2, 3, 5, 7]`
- `sobel_ksize`：`[3, 5, 7]`
- `threshold`：`[0.005, 0.01, 0.02, 0.05]`

#### 4.4.4 实验结果

每张图都生成四张合并图：

- `k_sweep.jpg`
- `block_size_sweep.jpg`
- `sobel_ksize_sweep.jpg`
- `threshold_sweep.jpg`

示例（以 `chessboard` 为例）：

- ![k_sweep.jpg](https://pic.zylona.site/i/2026/03/08/69acfdb831436.jpg)
- ![block_size_sweep.jpg](https://pic.zylona.site/i/2026/03/08/69acfe2138aa0.jpg)
- ![sobel_ksize_sweep.jpg](https://pic.zylona.site/i/2026/03/08/69acfdd4ca520.jpg)
- ![threshold_sweep.jpg](https://pic.zylona.site/i/2026/03/08/69acfe204db4f.jpg)

平均角点数量（对全部图片取均值）如下：

`k` 

| k | 平均角点数 |
|---:|---:|
| 0.02 | 21441.83 |
| 0.04 | 18346.67 |
| 0.06 | 16093.83 |
| 0.08 | 14264.67 |

`blockSize` 

| blockSize | 平均角点数 |
|---:|---:|
| 2 | 18346.67 |
| 3 | 21558.33 |
| 5 | 34553.67 |
| 7 | 48354.83 |

`sobel_ksize` 

| sobel_ksize | 平均角点数 |
|---:|---:|
| 3 | 18346.67 |
| 5 | 17738.67 |
| 7 | 17780.00 |

`threshold` 

| threshold | 平均角点数 |
|---:|---:|
| 0.005 | 26100.50 |
| 0.01 | 18346.67 |
| 0.02 | 11646.83 |
| 0.05 | 5135.50 |

#### 4.4.5 原因分析

1. `k` 增大时角点数量减少：
响应函数 `R = det(M) - k*trace(M)^2` 中，`k` 越大，惩罚项越强，弱角点更容易被抑制。

2. `blockSize` 增大时角点数量增加（本实验配置下）：
更大的邻域统计会使响应区域扩展，阈值筛选下通过的像素变多，表现为角点更“粗更密”。

3. `sobel_ksize` 对数量影响相对较小：
导数核变大主要改变局部平滑与梯度估计细节，整体趋势不如阈值和 blockSize 明显。

4. `threshold` 对结果最敏感：
阈值越低，保留响应越多；阈值越高，弱响应被快速剔除，角点数量显著下降。

#### 4.4.6 结论

- 在本作业实现中，影响强度大致为：`threshold`、`blockSize` > `k` > `sobel_ksize`。
- 若目标是稳定且不过密的角点，建议优先调 `threshold`，再调 `k` 与 `blockSize`。
- 报告展示时应结合可视化与数量统计，避免只看单一指标。


## 5. 结果讨论

综合实验结果，可以总结 Harris 算法的特点。

### 5.1 优点

1. 对几何结构敏感

Harris 对以下结构检测效果很好：

- 棋盘格
- 建筑物
- 门框
- 角落

原因：

角点在两个方向都有明显梯度变化。

------

2. 计算效率高

Harris 算法主要由以下操作组成：

```
卷积
矩阵运算
简单代数计算
```

复杂度较低。

因此在早期计算机视觉中应用非常广泛。

------

3. 稳定性较好

Harris 对：

- 小尺度噪声
- 光照变化

具有一定鲁棒性。

------

### 5.2 缺点

1. 不具备尺度不变性

Harris 只能检测 **固定尺度角点**。

如果图像缩放：

```
角点可能消失
```

解决方法：

```
Harris-Laplace
SIFT
```

------

2. 对纹理区域敏感

复杂纹理可能产生误检。

------

3. 参数依赖较强

算法结果对以下参数敏感：

- k
- threshold
- window size

需要经验调整。

## 6. 总结

本实验实现并验证了 **Harris 特征点检测算法**。

实验结果表明：

1. Harris 能有效检测几何角点
2. 对人工结构表现良好
3. 在自然纹理中表现较弱

同时通过与 OpenCV 对比，验证了手动实现的正确性。

Harris 算法虽然较为经典，但仍然是许多现代特征检测算法的基础。