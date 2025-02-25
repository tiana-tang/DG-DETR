"""
os: 用于处理文件路径和文件名操作。
time: 用于计算程序运行时间。
cv2: OpenCV 库，用于图像处理。
models: Django 提供的模块，用于定义数据库模型。
format_html: Django 提供的工具，用于生成 HTML 格式内容。
ContentFile: 用于创建文件内容的 Django 工具。
InMemoryUploadedFile: 表示用户上传的内存文件对象。
Detector: 自定义类，用于处理图像实例分割。
Record: 引用了历史记录模型。
get_current_user: 从中间件中获取当前操作用户
"""

import os
import time
import cv2
from django.db import models
from django.utils.html import format_html
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from model.detector import Detector
from history.models import Record
from middleware.current_user import get_current_user

# Create your models here.
# 辅助类：计算操作消耗的时间（以毫秒为单位）
class CostTime:
    def __init__(self):
        self.init_time = 0 

    def set_init_time(self):
        #记录开始时间。
        self.init_time = time.time()

    def get_cost_time(self):
        """计算程序运行消耗的时间.

        Returns:
            float: 程序运行消耗的时间(ms), 保留2位小数.
        """
        cur_time = time.time()
        return round((cur_time - self.init_time) * 1000, 2)


# 全局变量
# 全局实例，调用 Detector 类用于执行实例分割任务
detector = Detector()
# cost_timer: 全局实例，用于测量分割任务的耗时
cost_timer = CostTime()

# 定义模型
class Image(models.Model):
    image = models.ImageField(
        verbose_name="原始图像", upload_to="images/src/", blank=True, null=True
    )
    out = models.ImageField(
        verbose_name="分割结果",
        upload_to="images/out/",
        blank=True,
        null=True,
        editable=False,
    )
    mask = models.ImageField(
        verbose_name="掩码图像",
        upload_to="images/mask/",
        blank=True,
        null=True,
        editable=False,
    )
    create_time = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    update_time = models.DateTimeField(verbose_name="更新时间", auto_now=True)
    # 原始图像展示
    def image_display(self):
        return format_html("<img src={} width=400 height=300>", self.image.url)

    image_display.short_description = "原始图像"
    # 分割结果展示
    def out_display(self):
        basename = os.path.basename(
            self.image.name
        )  # 注: 原始图像名中最好不要有"-"或"_"符号, 否则可能出现BUG.
        if (
            self.out.name is None
            or len(self.out.name) == 0
            or os.path.basename(self.out.name).split("-")[0] != basename.split(".")[0]
        ):
            cost_timer.set_init_time()
            # 生成实例分割结果图像
            out_name = basename.split(".")[0] + "-result.png"
            _, out = cv2.imencode(
                ".png", detector.get_result(self.image.path)
            )  # 输出值的数据结构为: (Bool, Array)
            out_content = ContentFile(out)
            self.out.save(out_name, out_content, save=True)
            # 生成实例分割掩码图像
            mask_name = basename.split(".")[0] + "-mask.png"
            _, mask = cv2.imencode(
                ".png", detector.get_mask()
            )  # 输出值的数据结构为: (Bool, Array)
            mask_content = ContentFile(mask)
            self.mask.save(mask_name, mask_content, save=True)
            # 生成历史记录
            cur_user = get_current_user()
            record = Record(
                image=basename,
                inference_time=cost_timer.get_cost_time(),
                user_id=cur_user.id,
                user=cur_user.username,
            )
            record.save()
        return format_html("<img src={} width=400 height=300>", self.out.url)

    out_display.short_description = "分割结果"
    # 掩码图像展示
    def mask_display(self):
        return format_html("<img src={} width=400 height=300>", self.mask.url)

    mask_display.short_description = "掩码图像"

    # 保存
    def save_upload_image(self, upload: InMemoryUploadedFile):
        """保存上传的奶山羊图像并进行实例分割.

        Args:
            upload (InMemoryUploadedFile): 奶山羊图像.
        """
        self.image.save(upload.name, upload.file, save=True)

    class Meta:
        verbose_name = "奶山羊图像"
        verbose_name_plural = "奶山羊图像"

    def __str__(self) -> str:
        return self.image.url
