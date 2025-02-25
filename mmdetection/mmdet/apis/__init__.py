# Copyright (c) OpenMMLab. All rights reserved.
"""
async_inference_detector：用于异步推理图像，支持并行化推理，提高效率。
inference_detector：用于同步推理图像或图像列表。
init_detector：根据配置文件和权重初始化目标检测模型。
show_result_pyplot：用于可视化推理结果,将检测框和分割结果叠加到图像上。
multi_gpu_test：用于在多 GPU 环境下测试模型性能。
single_gpu_test：用于在单 GPU 环境下测试模型性能。
get_root_logger：获取训练日志记录器。
init_random_seed：初始化随机种子，用于确保实验结果的可复现性。
set_random_seed：设置随机种子，控制训练过程中的随机性。
train_detector：启动目标检测模型的训练过程。
"""
from .inference import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_detector)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_detector', 'init_detector',
    'async_inference_detector', 'inference_detector', 'show_result_pyplot',
    'multi_gpu_test', 'single_gpu_test', 'init_random_seed'
]
