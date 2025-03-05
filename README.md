# 🐐 Dairy Goat Instance Segmentation System

> **A web-based instance segmentation system for dairy goat images, built with Django & MMDetection.**

---

## 📌 Table of Contents
- [🌟 System Introduction](#-system-introduction)
- [🛠️ Code Structure](#-code-structure)
- [🛠️ Environment Setup](#-environment-setup)
  - [Installing MMDetection](#installing-mmdetection)
  - [Installing Django and Dependencies](#installing-django-and-dependencies)
- [📎 Dataset Download](#-dataset-download)
- [🚀 Usage Instructions](#-usage-instructions)
  - [Starting the Web Server](#starting-the-web-server)
  - [Model Training and Inference](#model-training-and-inference)
- [📝 Citation](#-citation)
- [📞 Contact](#-contact)

---

## 🌟 **System Introduction**
This system is a **B/S architecture-based** dairy goat image instance segmentation system designed to:
- **Efficiently process dairy goat image segmentation tasks**
- **Optimize intelligent agricultural management**
- **Support web-based operations and data visualization**

### **🔹 System Architecture**
```
User (Browser)  ↔  Django (Backend)  ↔  MMDetection (AI Model)  ↔  Training Data (COCO Format)
```
> 📏 **Frontend** uses Django admin, while the **backend** integrates MMDetection for instance segmentation.

---

## 🛠️ **Code Structure**
The project is structured as follows:
```
code/
 ├── mmdetection/            # Object detection and instance segmentation models (configurations and training scripts)
 │   ├── configs/            # Training configuration files
 │   ├── tools/              # Training and inference scripts
 │   ├── .....               # Other files
 │   ├── README.md           # Detailed usage instructions for this module
 │
 ├── segmentation_system/    # Web-based instance segmentation system (Django-based)
 │   ├── models/             # Business logic and database models
 │   ├── instance_segmentation    # Django configuration and system implementation
 │   ├── .....               # Other files
 │   ├── README.md           # Detailed usage instructions for this module
 │
 ├── README.md               # This file
```
> 📄 **For detailed usage guidelines, refer to** `mmdetection/README.md` and `segmentation_system/README.md`.

---

## 🛠️ **Environment Setup**
### **🔹 Installing MMDetection**
#### **1️⃣ Install PyTorch**
First, install PyTorch (with GPU acceleration support):
```shell
conda create --name dg-detr python=3.8 -y
conda activate dg-detr
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

#### **2️⃣ Install MMDetection**
```shell
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.0
```
Then install **MMDetection**:
```shell
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
# "-v" enables verbose output
# "-e" installs in editable mode, so local modifications are applied without reinstallation.
```

---

### **🔹 Installing Django and Dependencies**
#### **1️⃣ Install Django-related dependencies**
```shell
cd segmentation_system
pip install -r requirements.txt
```
#### **2️⃣ Initialize the Database**
```shell
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
# Follow the prompts to create an admin account
```

---

## 📎 **Dataset Download**
This project uses a **COCO format dataset** for training, available in the `Instance Segmentation` folder on GitHub.
The folder includes training, validation, and test sets.
📎 **[GitHub Dataset Link](https://github.com/tiana-tang/DiaryGoatMVT.git)**  

📄 **For a detailed dataset description, refer to the `README.md` inside the `Instance Segmentation` folder.**

---

## 🚀 **Usage Instructions**
### **🔹 Starting the Web Server**
```shell
cd segmentation_system
python manage.py runserver 8080
# To use a different port, e.g., 8006:
# python manage.py runserver 8006
```
Then open the following in your browser:
```
http://127.0.0.1:8080/admin/
```
📄 **For detailed web system usage, refer to** `segmentation_system/README.md`.

---

### **🔹 Model Training and Inference**
#### **1️⃣ Train the Model**
If you want to **retrain** the dairy goat segmentation model:
```shell
cd mmdetection
python tools/train.py configs/your_model_config.py
```

#### **2️⃣ Run Inference**
If you already have a trained model `best.pth`, run inference as follows:
```shell
cd mmdetection
python tools/test.py configs/your_model_config.py work_dirs/best.pth --eval bbox segm
```
📄 **For detailed training and inference steps, refer to** `mmdetection/README.md`.

---

## 📝 **Citation**
If you find our work useful in your research, please cite:

```bibtex
@article{Bai2025enhanced,
  title = {Enhanced Dairy Goat Instance Segmentation via Multi-Scale
Deformable Transformer},
  author = {Zihan Bai, Jinglei Tang, Xue Zhang, Hao Rong, Xianglong Pei and Yawei Ding},
  journal = {The Visual Computer},
  year = {2025},
  doi = {Your DOI}
}
```

---

## 📞 **Contact**
- **Email**: tangjinglei@nwsuaf.edu.cn
- **GitHub**: [DG-DETR](https://github.com/tiana-tang/DG-DETR.git)

