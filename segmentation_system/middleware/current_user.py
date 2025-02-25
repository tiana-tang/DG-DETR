from threading import local
from django.utils.functional import SimpleLazyObject
from django.utils.deprecation import MiddlewareMixin

_user = local()


class CurrentUserMiddleware(MiddlewareMixin):
    """当前用户信息获取中间件"""

    def process_request(self, request):
        _user.value = request.user


def get_current_user() -> SimpleLazyObject:
    """获取当前用户信息

    Returns:
        SimpleLazyObject: 当前用户对象.
    """
    return _user.value
