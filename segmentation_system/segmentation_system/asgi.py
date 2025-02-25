"""
ASGI config for segmentation_system project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
# """
#  Django ASGI �����ļ��������� ASGI��Asynchronous Server Gateway Interface�� ģʽ������ Django ��Ŀ��
# ����Ҫ������Ϊ��Ŀ�ṩһ�� ASGI Ӧ��ʵ������֧���첽Э�飨�� HTTP/2 �� WebSocket����������
# ����������Ŀ����ڵ� application���� ASGI ���������� Daphne �� Uvicorn�����ã��Ӷ����� Django ��Ŀ
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "segmentation_system.settings")

application = get_asgi_application()
