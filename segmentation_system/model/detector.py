from pathlib import Path 
# Path: 用于处理文件路径，提供比字符串更灵活的路径操作
import matplotlib
matplotlib.use("Agg")
# matplotlib 数据可视化库，通过 matplotlib.use("Agg") 适合无显示器的服务器环境，只输出静态图像文件
import mmcv 
# mmcv: 用于图像处理和机器学习的工具库，支持常用的深度学习框架
import numpy as np
# numpy: 常用的数值计算库
from mmdet.apis import inference_detector, init_detector
# mmdet.apis: MMDetection 提供的 API，inference_detector 和 init_detector 分别用于模型推理和初始化。
BASE_DIR = Path(__file__).resolve().parent.parent
# Path(__file__).resolve()：获取当前文件的绝对路径；.parent.parent：获取当前文件两级父目录路径，作为项目根目录
# 也就是整个系统的文件夹目录：SEGMENTATION_SYSTEM
CONFIG = str(BASE_DIR / "model/configs/model_config.py")
CHECKPOINT = str(BASE_DIR / "model/checkpoints/best.pth")
# 模型配置文件和莫模型文件的路径，通过拼接 BASE_DIR 和相对路径生成
DEVICE = "cuda:0"
# 表示使用第一块GPU
SCORE_THR = 0.3
# SCORE_THR: 置信度阈值，低于该值的检测结果会被过滤掉

# 定义一个用于实例分割的检测器类
class Detector:
    """奶山羊图像实例分割检测器"""

    # 初始化函数，构造函数，接受模型配置文件、权重文件、运行设备和置信度阈值为参数，支持自定义。
    def __init__(
        self, config=CONFIG, checkpoint=CHECKPOINT, device=DEVICE, score_thr=SCORE_THR
    ) -> None:
        # 模型配置文件 ：segmentation_system\model\configs\model_config.py
        self.config = config
        # 模型文件：segmentation_system\model\checkpoints/best.pth
        self.checkpoint = checkpoint
        # 用于检测的设备
        self.device = device
        # 置信阈值
        self.score_thr = score_thr
        # 检测模型
        # 使用 init_detector 初始化模型。这里模型会加载权重文件到指定设备
        self.model = init_detector(config, checkpoint, device)
        # 原始图像路径
        self.image = None
        # 检测结果
        self.detect_result = None

    # 检测方法
    def detect(self, image):
        """生成检测结果.

        Args:
            image (str): 原始图像路径.

        Returns:
            tuple: 检测结果.
        """
        # 将输入的图像路径赋值给实例属性 self.image
        self.image = image
        # 使用模型对输入的图像进行推理，获取检测结果
        # inference_detector 是来自 mmdet.apis 的函数，用于执行模型的推理过程
        # self.model: 之前在构造函数中初始化的检测模型
        # self.detect_result 是 detect 方法的返回值，包含：bbox_result: 检测到的边界框 [x1, y1, x2, y2, score]。
        # segm_result: 包含所有检测到的实例分割掩码。
        self.detect_result = inference_detector(self.model, image)
        return self.detect_result

# 方法说明：接收图像路径，生成并返回带实例分割结果的图像
    def get_result(self, image):
        """生成实例分割结果图像.

        Args:
            image (str):原始图像路径.

        Returns:
            numpy.ndarray: 实例分割结果图像.
        """
        # 得到检测结果
        result = self.detect(image)
        # 使用 model.show_result 将检测结果绘制到图像上并返回
        return self.model.show_result(image, result)

    # 方法说明：返回一个掩码图像，每个分割的实例对应图像中的白色区域
    def get_mask(self):
        """生成实例分割掩码图像.

        Returns:
            numpy.ndarray: 实例分割掩码图像.
        """
        # 在业务逻辑中, get_mask必须在get_result之后执行, 因此直接利用已有self.detect_result.
        # bbox_result 是 inference_detector 返回的检测结果之一，表示检测到的边界框信息
        bbox_result, segm_result = self.detect_result
        # 将所有类别的边界框结果按行方向堆叠成一个二维数组，格式为 [x1, y1, x2, y2, score]
        bboxes = np.vstack(bbox_result)
        segms = None
        # 如果分割结果存在，将其展开为三维数组，每一层为一个实例的分割掩码
        # mmcv.concat_list:将嵌套的分割掩码列表合并为一个单层列表
        # np.stack:将每个掩码堆叠为一个三维数组，形状为 (实例数, 高度, 宽度)
        if segm_result is not None:
            segms = np.stack(mmcv.concat_list(segm_result), axis=0)

        # 仅保留大于置信阈值的结果
        if self.score_thr > 0:
            # 从 bboxes 的每一行提取最后一列（置信度分数），返回一个一维数组，其中包含每个边界框的置信度分数
            scores = bboxes[:, -1]
            # 选出大于置信度阈值的结果
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            if segms is not None:
                # 通过 segms[inds, ...] 筛选出满足条件的分割掩码，...: 表示对后三个维度（高度和宽度）保留所有数据
                segms = segms[inds, ...]

        # 创建掩码图像
        # 使用原始图像的高度和宽度，创建一个全黑的掩码图像 mask
        # mmcv.imread 是 MMCV 提供的一个图像读取函数，用于加载图像文件为数组格式，
        # astype(np.uint8)将图像数组的像素值强制转换为 uint8 类型（无符号8位整数，取值范围 0~255）
        # .shape获取图像数组的形状信息 (高度, 宽度, 通道数)
        # _: 通道数，使用下划线表示该值在当前代码中不会被用到
        h, w, _ = mmcv.imread(self.image).astype(np.uint8).shape
        # np.zeros 是 NumPy 提供的函数，用于创建一个指定形状的数组，所有元素的初始值为 0
        mask = np.zeros(shape=(h, w, 1), dtype=np.uint8)
        if segms is not None:
            # instance它是分割结果的二维数组（单个实例的分割掩码）
            for instance in segms:
                # astype(np.bool_) 的作用:将 instance 的像素值转换为布尔类型
                mask[instance.astype(np.bool_)] = 255  # 白色部分表示分割的实例
        return mask
