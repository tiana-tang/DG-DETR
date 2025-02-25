"""
WSGI config for segmentation_system project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""
# Django 项目的 WSGI 配置文件，其主要功能是为项目提供一个 WSGI 应用对象，以支持 WSGI（Web Server Gateway Interface） 协议。
# WSGI 是 Python Web 应用和 Web 服务器之间的通信标准。
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "segmentation_system.settings")

application = get_wsgi_application()
