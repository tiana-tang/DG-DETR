from django.contrib import admin
from django.http.request import HttpRequest
from .models import Record

# Register your models here.
@admin.register(Record)
class RecordAdmin(admin.ModelAdmin):
    list_display = ["id", "image", "inference_time", "user_id", "user", "create_time"]
    list_editable = []
    list_filter = ["user"]
    list_per_page = 20

    def has_add_permission(self, request: HttpRequest) -> bool:
        # 屏蔽增加记录按钮
        return False
