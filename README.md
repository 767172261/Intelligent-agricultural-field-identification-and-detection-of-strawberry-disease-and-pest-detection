智能化农业领域识别检测 草莓病虫害检测数据集，配合 RT-Detr 模型可实现对草莓病害检测数据集的识别
@[toc]
![根据您提供的信息和图片中的评估结果，以下是整理后的“草莓病虫害检测数据集”表格，包含了类别、图像数量、实例数量以及训练评估指标等详细信息：](https://i-blog.csdnimg.cn/direct/fdecea4ed3634b2d9b37b199914d1013.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8097111abe0141a7824025e9c2fa656f.png)


```
rt detr-r18 summary: 300 layers, 19879464 parameters, 0 gradients, 57.0 GFLOPs

Class          Images Instances Box(P)    R       mAP50   mAP50-95
all            578    1403     0.928     0.918   0.945   0.795
Angular Leafspot 578    129      0.963     0.803   0.919   0.776
Anthracnose Fruit Rot 578    21       0.949     1       0.995   0.771
Gray Mold      578    128      0.953     0.945   0.96    0.73
Leaf Spot      578    580      0.947     0.948   0.983   0.915
Powdery Mildew Fruit 578    86       0.873     0.882   0.864   0.711
Powdery Mildew Leaf 578    459      0.883     0.932   0.948   0.866

Speed: 0.3ms preprocess, 4.7ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to runs/train/rtdetr-r18
```

---

### **草莓病虫害检测数据集**

| **项目** | **内容** |
|----------|----------|
| **数据集名称** | 草莓病虫害检测数据集 |
| **图像总数** | 9,080 张 |
| **训练图像数** | 7,925 张 |
| **验证图像数** | 578 张 |
| **测试图像数** | 未指定（可选） |
| **标注格式** | YOLO `.txt` 格式 |
| **类别数量** | 6 类 |
| **类别名称** | Angular Leafspot, Anthracnose Fruit Rot, Gray Mold, Leaf Spot, Powdery Mildew Fruit, Powdery Mildew Leaf |
| **训练mAP@50** | 94.5% |

---


```yaml
train: # train images (relative to 'path') 7925 images
- train/images

val: # val images (relative to 'path') 578 images
- valid/images

test: # test images (optional)
- test/images

# ['角斑病', '炭疽果腐病', '灰霉病', '叶斑病', '果实白粉病', '叶片白粉病']

nc: 6
names: ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Gray Mold', 'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf']
```


### **数据集划分详情**

| **划分类型** | **图像路径** | **图像数量** |
|--------------|--------------|--------------|
| **训练集**   | `train/images` | 7,925 张 |
| **验证集**   | `valid/images` | 578 张 |
| **测试集**   | `test/images` | 未指定（可选） |

---

### **类别信息**

| **类别编号** | **类别名称** | **中文名称** |
|--------------|--------------|--------------|
| 0            | Angular Leafspot | 角斑病 |
| 1            | Anthracnose Fruit Rot | 炭疽果腐病 |
| 2            | Gray Mold | 灰霉病 |
| 3            | Leaf Spot | 叶斑病 |
| 4            | Powdery Mildew Fruit | 果实白粉病 |
| 5            | Powdery Mildew Leaf | 叶片白粉病 |

---

### **性能指标**

| **指标** | **数值** |
|----------|----------|
| **mAP@50** | 94.5% |

---

### **数据集特点总结**

- ✅ **大规模数据**：共计 9,080 张图像，其中训练集 7,925 张，验证集 578 张。
- ✅ **已划分好**：数据集已预先划分为训练集和验证集，方便直接用于模型训练与评估。
- ✅ **YOLO格式**：标注文件为 YOLO `.txt` 格式，可以直接用于 YOLO 系列模型的训练。


---






---

## ✅ 一、项目结构建议

```
strawberry_disease_project/
│
├── datasets/
│   └── strawberry_diseases/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── strawberry_diseases.yaml
│
├── models/
│   └── rtdetr-r18.pt          # RT-Detr 预训练权重（或自行下载）
│
├── runs/                      # 自动保存训练结果
│
├── train.py                   # 训练脚本
├── detect.py                  # 推理脚本
├── val.py                     # 评估脚本
├── install_env.py             # 环境安装脚本
└── README.md
```

---

## ✅ 二、环境搭建全流程

### 1. 安装 CUDA 驱动（GPU 支持）

- 确保你的 NVIDIA 显卡驱动支持 CUDA。
- 打开命令行，输入：
  ```bash
  nvidia-smi
  ```
  若显示 GPU 信息（如显存、驱动版本），则 CUDA 驱动正常。

> 推荐 CUDA 版本：11.8 或 12.1（与 PyTorch 兼容）

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装后验证：
  ```bash
  conda --version
  ```

---

### 3. 创建 Python 虚拟环境并安装依赖

#### `install_env.py`

```python
# install_env.py
import subprocess
import sys

def install_packages():
    # 创建虚拟环境
    subprocess.run([
        "conda", "create", "-n", "strawberry_yolo", "python=3.9", "-y"
    ], shell=True)

    # 激活并安装包
    commands = [
        "conda activate strawberry_yolo",
        "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html",
        "pip install ultralytics opencv-python PyQt5 matplotlib pillow"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    install_packages()
```

激活环境：
```bash
conda activate strawberry_yolo
```

---

## ✅ 三、数据集配置文件 `strawberry_diseases.yaml`

```yaml
# strawberry_diseases.yaml
train: ../datasets/strawberry_diseases/images/train
val: ../datasets/strawberry_diseases/images/val
test: ../datasets/strawberry_diseases/images/test

nc: 6
names: ['Angular Leafspot', 'Anthracnose Fruit Rot', 'Gray Mold', 'Leaf Spot', 'Powdery Mildew Fruit', 'Powdery Mildew Leaf']
```

> ✅ `nc: 6` 表示 6个类别，请根据你的实际类别顺序填写 `names`。

---

## ✅ 四、训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO

# 1. 加载官方预训练模型（推荐 yolov8n 或 yolov8s）
model = YOLO('models/rtdetr-r18.pt')  # 可换为其他预训练权重

# 2. 开始训练
results = model.train(
    data='datasets/strawberry_diseases/strawberry_diseases.yaml',  # 数据集配置
    epochs=200,               # 训练轮数
    imgsz=640,                # 输入尺寸
    batch=16,                 # 批次大小（根据显存调整，9080张图可用16）
    name='strawberry_rtdetr_640',  # 实验名称
    device='cuda',            # 使用 GPU
    workers=8,                # 数据加载线程
    optimizer='AdamW',        # 优化器
    lr0=0.001,                # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,              # 早停：50轮无提升则停止
    project='runs/strawberry',   # 保存路径
    amp=True,                 # 混合精度训练（加快速度）
    close_mosaic=30,          # 关闭 Mosaic 数据增强的轮数
    cache='disk'              # 缓存图像到磁盘，加快训练
)

# 3. 打印结果
print("训练完成！")
print(results)
```

### 🚀 训练命令：
```bash
conda activate strawberry_yolo
python train.py
```

训练完成后，最佳模型保存在：
```
runs/strawberry/strawberry_rtdetr_640/weights/best.pt
```

---

## ✅ 五、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/strawberry/strawberry_rtdetr_640/weights/best.pt')

conf_threshold = 0.3

# --- 1. 检测单张图片 ---
def detect_image():
    results = model.predict(
        source='datasets/strawberry_diseases/images/test/img001.jpg',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='image_result'
    )
    print(f"图片检测完成，结果保存在 runs/detect/image_result/")

# --- 2. 检测整个文件夹 ---
def detect_folder():
    results = model.predict(
        source='datasets/strawberry_diseases/images/test/',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='batch_result'
    )
    print(f"批量检测完成，结果保存在 runs/detect/batch_result/")

# --- 3. 实时摄像头检测 ---
def detect_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_threshold)
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('Strawberry Disease Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 4. 检测视频文件 ---
def detect_video():
    results = model.predict(
        source='test_video.mp4',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='video_result'
    )

if __name__ == "__main__":
    detect_folder()  # 可切换为 detect_image(), detect_camera(), detect_video()
```

---

## ✅ 六、评估代码：`val.py`

评估模型在验证集上的性能（mAP, Precision, Recall 等）

```python
# val.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/strawberry/strawberry_rtdetr_640/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='datasets/strawberry_diseases/strawberry_diseases.yaml',
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device='cuda'
)

# 打印关键指标
print("类别数量:", metrics.box.nc)
print("各类别 mAP50:")
for i, name in enumerate(metrics.names):
    print(f"  {name}: {metrics.box.map50_per_class[i]:.4f}")

print("\n整体性能:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
print(f"F1 Score: {metrics.box.f1.mean():.4f}")

# 混淆矩阵等结果保存在 runs/val/strawberry_rtdetr_640/
```

---

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/462f00dc08b44e57b7a0315146c9ef7e.png)


---

## ✅ 八、使用建议与优化

| 问题 | 建议 |
|------|------|
| 数据量大（9080张） | 启用数据增强（YOLOv8 默认开启） |
| 类别不平衡 | 检查 `labels/` 中各类别数量，可使用 `labelimg` 或 `yolo utils` 分析 |
| 检测框抖动 | 推理时使用 `iou=0.45`, `conf=0.4` 更稳定 |
| 显存不足 | 改用 `yolov8n.pt`，`batch=8` 或 `4` |

---

## ✅ 九、一键运行脚本（可选）

```python
# run_all.py
import subprocess

subprocess.run(["python", "train.py"])
subprocess.run(["python", "val.py"])
subprocess.run(["python", "detect.py"])
```


---

