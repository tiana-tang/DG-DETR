"""
WSGI config for segmentation_system project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""
# Django ��Ŀ�� WSGI �����ļ�������Ҫ������Ϊ��Ŀ�ṩһ�� WSGI Ӧ�ö�����֧�� WSGI��Web Server Gateway Interface�� Э�顣
# WSGI �� Python Web Ӧ�ú� Web ������֮���ͨ�ű�׼��
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "segmentation_system.settings")

application = get_wsgi_application()
