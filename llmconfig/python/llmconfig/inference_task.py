import base64
from llmconfig.logger import init_logger
import types

import numpy as np
import requests
import websockets
import asyncio
from PIL import Image
from io import BytesIO

image_tasks = ['image-cls', 'image-det', 'image-seg']
llm_tasks = ['llm-generate', 'vl-llm-generate']

logging = init_logger("llmconfig_logger")

class InferenceTask:
    def __init__(self, cfg: dict):
        if not 'classes' in cfg:
            raise RuntimeError('classes is missing in meta config')
        else:
            classes = cfg['classes']

        if not isinstance(classes, list) or not all(isinstance(cls, str) for cls in classes):
            raise RuntimeError(f'classes should be list of str, but got {type(classes).__name__}')

        self.classes = {index: elem for index, elem in enumerate(classes)}
        self.class_num = len(classes)


class ImageClsTask(InferenceTask):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def post_process(self, predicts):
        if not isinstance(predicts, list) or not all(
                isinstance(p, np.ndarray) and (p.dtype != np.float32 or p.dtype != np.float64) for p in predicts):
            raise RuntimeError(f"only list[numpy.array] with float is allowed, but got {type(predicts).__name__}")
        else:
            for p in predicts:
                if p.ndim != 2 or p.shape[1] < 2:
                    raise RuntimeError(f"only 2D(Nx2) numpy.array is allowed, but got {predicts.shape}")

        return [[{
            'class': self.classes[int(r[0])] if r[0] < self.class_num else f'unknown[{r[0]}]',
            'score': r[1]} for r in p] for p in predicts]


class ImageDetTask(InferenceTask):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def post_process(self, predicts):
        if not isinstance(predicts, list) or not all(
                isinstance(p, np.ndarray) and (p.dtype != np.float32 or p.dtype != np.float64) for p in predicts):
            raise RuntimeError(f"only list[numpy.array] with float is allowed, but got {type(predicts).__name__}")
        else:
            for p in predicts:
                if p.ndim != 2 or p.shape[1] < 6:
                    raise RuntimeError(f"only 2D(Nx6) numpy.array is allowed, but got {predicts.shape}")

        return [[{
            'class': self.classes[int(r[0])] if r[0] < self.class_num else f'unknown[{r[0]}]',
            'score': r[1],
            'xmin': r[2], 'xmax': r[3],
            'ymin': r[4], 'ymax': r[5]} for r in p] for p in predicts]


class LLMTask(InferenceTask):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

    def post_process(self, predicts):
        if isinstance(predicts, dict):
            if predicts.get("output", None) is not None and len(predicts["output"]) == 0:
                raise RuntimeWarning(f"output size is 0")
        return predicts


def get_img_handler(task, cfg: dict):
    if task == 'image-cls':
        return ImageClsTask(cfg)
    elif task == 'image-det':
        return ImageDetTask(cfg)
    elif task == 'llm-generate':
        return LLMTask(cfg)
    elif task == 'vl-lm-generate':
        return LLMTask(cfg)
    else:
        raise RuntimeError(f'unsupported task {task}, supported: {image_tasks}')


def decode_image(imgBytes):
    img = Image.open(BytesIO(imgBytes))
    if img.mode != 'RGB':
        raise RuntimeError(f'invalid image mode: {img.mode}, expects RGB')
    else:
        return np.array(img)


def parse_b64_image(imgB64):
    img = base64.b64decode(imgB64)
    return decode_image(img)


def download_image(url):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f'download image {url} failed: {resp.text}')

    return decode_image(resp.content)
