from django.db import models

# Create your models here.
class Feedback(models.Model):
    title = models.CharField(verbose_name="反馈意见", max_length=128)
    content = models.TextField(verbose_name="具体描述")
    type_choices = ((0, "使用问题"), (1, "改进建议"))
    feedback_type = models.IntegerField(
        verbose_name="反馈意见类型", choices=type_choices, default=0
    )
    create_time = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)
    update_time = models.DateTimeField(verbose_name="更新时间", auto_now=True)

    class Meta:
        verbose_name = "反馈意见"
        verbose_name_plural = "反馈意见"

    def __str__(self) -> str:
        return self.title
