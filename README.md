# 🐐 奶山羊图像实例分割系统

> **A web-based instance segmentation system for dairy goat images, built with Django & MMDetection.**

---

## 📌 目录 (Table of Contents)
- [🌟 系统介绍](#-系统介绍)
- [🛠️ 代码结构](#-代码结构)
- [🛠️ 环境搭建](#-环境搭建)
  - [安装 MMDetection](#安装-mmdetection)
  - [安装 Django 及依赖](#安装-django-及依赖)
- [📎 数据集下载](#-数据集下载)
- [🚀 使用说明](#-使用说明)
  - [启动 Web 服务器](#启动-web-服务器)
  - [模型训练与推理](#模型训练与推理)
- [📝 论文引用](#-论文引用)
- [📞 联系我们](#-联系我们)

---

## 🌟 **系统介绍**
本系统是一种基于 **B/S 架构** 的奶山羊图像实例分割系统，旨在：
- **高效处理奶山羊图像分割任务**
- **优化农业智能养殖的管理**
- **支持 Web 端操作、数据可视化**

### **🔹 系统架构**
```
用户（浏览器）  ↔  Django（后端）  ↔  MMDetection（AI模型）  ↔  训练数据（COCO格式）
```
> 📏 **前端** 采用 Django admin，**后端** 结合 MMDetection 进行实例分割。

---

## 🛠️ **代码结构**
项目的代码组织如下：
```
code/
 ├── mmdetection/            # 目标检测和实例分割模型（包括配置文件和训练代码）
 │   ├── configs/            # 训练配置文件
 │   ├── tools/              # 训练和推理脚本
 │   ├── .....               # 其他文件
 │   ├── README.md           # 该模块的详细使用说明
 │
 ├── segmentation_system/    # Web 端实例分割系统（基于 Django）
 │   ├── models/             # 业务逻辑和数据库模型
 │   ├── instance_segmentation    # Django文件配置，系统实现
 │   ├── .....               # 其他文件
 │   ├── README.md           # 该模块的详细使用说明
 │
 ├── README.md               # 本文件
```
> 📄 **详细的使用指南请参考** `mmdetection/README.md` 和 `segmentation_system/README.md`。

---

## 🛠️ **环境搭建**
### **🔹 安装 MMDetection**
#### **1️⃣ 安装 PyTorch**
请先安装 PyTorch（支持 GPU 加速）：
```shell
conda create --name dg-detr python=3.8 -y
# seg-sys可更换为任何喜欢的名字
conda activate dg-detr
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

#### **2️⃣ 安装 MMDetection**
```shell
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.0
```
然后安装 **MMDetection**：
```shell
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

---

### **🔹 安装 Django 及依赖**
#### **1️⃣ 安装 Django 相关依赖**
```shell
cd segmentation_system
pip install -r requirements.txt
```
#### **2️⃣ 初始化数据库**
```shell
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
# 按提示创建管理员账号
```

---

## 📎 **数据集下载**
本项目使用 **COCO 格式数据集** 进行训练，数据集存放于 GitHub 的 `Instance Segmentation` 文件夹下，
该文件夹包含训练集、验证集和测试集。
📎 **[GitHub 数据集链接](https://github.com/tiana-tang/DiaryGoatMVT.git)**  

📄 **数据集详细介绍** 请参考 `Instance Segmentation` 文件夹内的 `README.md`。

---

## 🚀 **使用说明**
### **🔹 启动 Web 服务器**
```shell
cd segmentation_system
python manage.py runserver 8080
# 如果想更改为其他端口，如8006，可执行：python manage.py runserver 8006
```
然后在浏览器访问：
```
http://127.0.0.1:8080/admin/
```
📄 **Web 端的详细使用指南请参考** `segmentation_system/README.md`。

---

### **🔹 模型训练与推理**
#### **1️⃣ 训练模型**
如果你希望 **重新训练** 奶山羊分割模型：
```shell
cd mmdetection
python tools/train.py configs/your_model_config.py
```

#### **2️⃣ 运行推理**
如果已有模型 `best.pth`，可以直接运行推理：
```shell
cd mmdetection
python tools/test.py configs/your_model_config.py work_dirs/best.pth --eval bbox segm
```
📄 **详细的训练和推理步骤请参考** `mmdetection/README.md`。

---

## 📝 **论文引用**
```bibtex
@article{your_paper_citation,
  title = {Dairy Goat Instance Segmentation System},
  author = {Your Name et al.},
  journal = {The Visual Computer},
  year = {2025},
  doi = {Your DOI}
}
```

---

## 📞 **联系我们**
- **邮箱**：tangjinglei@nwsuaf.edu.cn
- **GitHub**：https://github.com/tiana-tang/DG-DETR.git

