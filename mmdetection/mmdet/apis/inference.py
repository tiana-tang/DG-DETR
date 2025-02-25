# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


# 初始化目标检测模型
# 用于根据配置文件和检查点加载目标检测模型
# config 配置文件；checkpoint 权重文件；device：运行设备；cf_options:可选字典，用于在代码中动态修改配置文件中的设置
# 返回一个构建好的 PyTorch 模型对象
def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    # 读取配置文件
    """
    检查 config 的类型。
如果 config 是路径字符串或 Path 对象，则用 mmcv.Config.fromfile 加载配置文件。
如果 config 不是 mmcv.Config 类型，也不是路径类型，抛出异常。
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    # 修改配置文件内容：如果 cfg_options 不为空，用 merge_from_dict 方法将 cfg_options 中的参数覆盖到配置文件中
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
        # 删除预训练权重配置，如果模型配置中包含预训练权重（pretrained 或 init_cfg），将其设置为 None
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

        # 将模型的 train_cfg 设置为 None，明确指定模型处于推理模式
    config.model.train_cfg = None
    # 构建模型
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    # 加载权重
    """
    如果提供了 checkpoint（权重文件路径）：
使用 load_checkpoint 加载权重到模型中。
检查权重文件的元信息（meta）中是否包含类别名（CLASSES）。
如果存在，设置 model.CLASSES 为权重中的类别名。
如果不存在，警告用户并默认使用 COCO 数据集的类别。
    """
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
            # 保存配置到模型
    model.cfg = config  # save the config in the model for convenience
    # 将模型移动到指定的设备（如 GPU 或 CPU）
    model.to(device)
    # 切换到评估模式
    model.eval()
    # 返回初始化完成的模型对象,提供一个可直接用于推理或进一步测试的模型
    return model

# LoadImage 是一个简单的数据预处理模块，用于加载图像并将其相关信息存储到结果字典中
class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """
# 将输入的结果字典（results）中包含的图像文件加载到内存中，并记录相关的元数据信息
    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # 废弃警告
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        # 处理图像路径信息
        """
        检查 results['img'] 的类型是否为字符串。
如果是字符串，表示 results['img'] 是图像文件的路径：
设置 filename 和 ori_filename 为该路径。
如果不是字符串，则设置 filename 和 ori_filename 为 None。
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
            # 加载图像
        img = mmcv.imread(results['img'])
        # 更新结果字典
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


# 单图或批量图像的推理
# 是用于使用检测器模型对输入图像或图像列表进行推理的函数
def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
# 判断输入是否为批量图像
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
# 提取模型配置与设备信息
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
# 处理摄像头输入的特殊情况
    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
     
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
   #  替换数据加载管道
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
# 构造输入数据
    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
# 打包数据并分配到设备
    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    # 推理
    
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results

# 结果可视化
def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)
