# 🐐 奶山羊图像实例分割系统

> **A web-based instance segmentation system for dairy goat images, built with Django & MMDetection.**

---

## 📌 目录 (Table of Contents)
- [🌟 系统介绍](#-系统介绍)
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
> 📏 **前端** 采用 Django admin，**后端** 结合 MMDetection 进行实例分割。

---

## 🛠️ **环境搭建**
### **🔹 安装 MMDetection**
#### **1️⃣ 安装 PyTorch**
请先安装 PyTorch（支持 GPU 加速）：
```shell
conda create --name dg-detr python=3.8 -y
# dg-detr可更换为任何喜欢的名字
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
本项目使用 **COCO 格式数据集** 进行训练：
📎 **[GitHub 数据集链接](https://github.com/tiana-tang/DiaryGoatMVT.git)**  
其中 Instance Segmentation文件为模型的训练数据集，其中包含训练集、验证集和测试集。

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
对与具体在系统中的操作请参考segmentation_system/README.md
---

## 📝 **论文引用**
```bibtex
@article{Bai2025enhanced,
  title = {Enhanced Dairy Goat Instance Segmentation via Multi-Scale
Deformable Transformer},
  author = {Zihan Bai, Jinglei Tang, Xue Zhang, Hao Rong, Xianglong Pei and Yawei Ding},
  journal = {The Visual Computer},
  year = {2025},
  doi = {DOI}
}
```

---

## 📞 **联系我们**  
- **邮箱**：tangjinglei@nwsuaf.edu.cn 
- **GitHub**：https://github.com/tiana-tang/DG-DETR.git
