# 导入
"""
os: 处理文件路径和后缀。
zipfile: 用于将文件压缩为 ZIP 格式。
BytesIO: 用于操作内存中的二进制数据（避免创建临时文件）。
datetime: 用于生成带时间戳的文件名。
cv2: OpenCV，用于图像处理。
JsonResponse: 返回 JSON 格式的 HTTP 响应。
FileResponse: 返回文件的 HTTP 响应，支持下载。

admin: Django 管理后台的模块。
AjaxAdmin: SimpleUI 提供的增强功能，用于实现 Ajax 操作
"""
import os
import zipfile 
from io import BytesIO
from datetime import datetime
import cv2
from django.contrib import admin
from django.http import JsonResponse, FileResponse
from simpleui.admin import AjaxAdmin
from .models import Image

# 注册模型到 Admin 后台
# Register your models here.
# 使用 SimpleUI 的 AjaxAdmin 类，支持 Ajax 异步操作
@admin.register(Image)
class ImageAdmin(AjaxAdmin):
    list_display = (
        "id",
        "image_display",
        "out_display",
        "mask_display",
        "create_time",
        "update_time",
    )
    # 设置可点击跳转到详情页的字段。
    list_display_links = ("id", "image_display")
    # 每页显示的记录数，这里设置为 10 条。
    list_per_page = 10
    # 定义自定义的管理后台操作
    actions = ("upload_image", "download_image")

    # 上传
    def upload_image(self, request, queryset):
        # 获取上传文件
        upload = request.FILES["upload"]  # 同params中配置的key
        # 创建模型实例
        img = Image()
        # 调用模型方法保存文件
        img.save_upload_image(upload)
        return JsonResponse(
            data={"status": "success", "msg": f"文件 {upload.name} 上传成功!"}
        )

    # UI配置
    upload_image.short_description = "奶山羊图像上传"  # 按钮名称
    upload_image.type = "warning"  # 警告按钮配色
    upload_image.icon = "el-icon-upload"    # 按钮图标
    upload_image.enable = True
    upload_image.layer = {
        "params": [
            {"type": "file", "key": "upload", "label": "文件", "accept": "image/*"}
        ]
    }

    # 下载
    def download_image(self, request, queryset):
        download_io = BytesIO() # 初始化内存缓冲区
        zipname = f"download-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.zip"

        # 下载已选择的图像记录
        with zipfile.ZipFile(download_io, "w") as zf:
            for info in queryset.values():
                obj = Image.objects.get(id=info["id"])
                images = [obj.image, obj.out, obj.mask]  # 原图, 分割结果, 掩码图像
                for image in images:
                    data = cv2.imread(image.path)
                    postfix = os.path.splitext(obj.out.path)[-1]  # 文件后缀
                    _, data = cv2.imencode(postfix, data)
                    zf.writestr(image.url, data)
        download_io.seek(0)  # 需要要将指针指向内存的开始位置

        response = FileResponse(download_io,filename=zipname, as_attachment=True)
        return response

    download_image.short_description = "分割结果下载"
    download_image.type = "success"  # 成功按钮配色
    download_image.icon = "fa-solid fa-cloud-arrow-down"
    download_image.enable = True
