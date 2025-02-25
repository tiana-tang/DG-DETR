from django.contrib import admin
from import_export import resources
from import_export.admin import ExportActionModelAdmin
from .models import Feedback

# Register your models here.
class ProxyResource(resources.ModelResource):
    class Meta:
        model = Feedback


@admin.register(Feedback)
class FeedbackAdmin(ExportActionModelAdmin):
    resource_class = ProxyResource
    list_display = (
        "id",
        "title",
        "content",
        "feedback_type",
        "create_time",
        "update_time",
    )
    search_fields = ("title", "content")
    list_per_page = 20
