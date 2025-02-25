"""
ASGI config for segmentation_system project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
# """
#  Django ASGI 配置文件，用于在 ASGI（Asynchronous Server Gateway Interface） 模式下运行 Django 项目。
# 其主要作用是为项目提供一个 ASGI 应用实例，以支持异步协议（如 HTTP/2 和 WebSocket）的请求处理。
# 它定义了项目的入口点 application，供 ASGI 服务器（如 Daphne 或 Uvicorn）调用，从而运行 Django 项目
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "segmentation_system.settings")

application = get_asgi_application()
