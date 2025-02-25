from django.db import models

# Create your models here.
class Record(models.Model):
    image = models.CharField(
        verbose_name="图像名称",
        max_length=128,
        blank=False,
        null=False,
        editable=False,
        help_text="已上传的奶山羊图像",
        db_index=True,
    )
    inference_time = models.FloatField(
        verbose_name="推理耗时(ms)", blank=False, null=False, editable=False
    )
    user_id = models.IntegerField(
        verbose_name="用户id",
        blank=False,
        null=False,
        editable=False,
        help_text="执行操作的用户id",
    )
    user = models.CharField(
        verbose_name="用户名称",
        max_length=128,
        blank=False,
        null=False,
        editable=False,
        help_text="执行操作的用户",
        db_index=True,
    )
    create_time = models.DateTimeField(verbose_name="创建时间", auto_now_add=True)

    class Meta:
        verbose_name = "操作记录"
        verbose_name_plural = "操作记录"

    def __str__(self) -> str:
        return f"{self.image} by {self.user}"
