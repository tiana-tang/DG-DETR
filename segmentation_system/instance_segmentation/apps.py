# 定义一个名为 instance_segmentation 的 Django 应用的配置信息。
# AppConfig 是 Django 提供的类，用于管理应用的初始化、配置和元信息。
from django.apps import AppConfig


class InstanceSegmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "instance_segmentation"
    verbose_name = "图像实例分割"
