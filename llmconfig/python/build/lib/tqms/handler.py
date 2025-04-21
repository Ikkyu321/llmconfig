import os
import json
import importlib

from inference_task import *


class Handler:
    processorTypes = ['py']

    def __init__(self):
        print(f'working dir: {os.getcwd()}')
        self.meta = self.load_meta()
        # check task type
        self.task_type = self.read_meta_field('taskType')

        if self.task_type in image_tasks:
            self.postH = get_img_handler(self.task_type, self.meta)
        elif self.task_type in llm_tasks:
            self.postH = get_img_handler(self.task_type, self.meta)
        else:
            raise RuntimeError(f'unsupported task type {self.task_type}')

        # check processor
        self.processor_type = self.read_meta_field('processorType', Handler.processorTypes)
        self.processor_name = ""
        if self.task_type == "image-det":
            self.processor_name = "Detect"
        if self.task_type == "llm-generate" or self.task_type ==  "vl-llm-generate":
            self.processor_name = "LLM_Chat"

        self.batch_size = self.read_meta_field('batchSize')

        # load handler
        print('load inference handler')
        self.handler = self.load_handler()

    def __call__(self, inputs, params):
        return self.postH.post_process(self.handler.process(inputs, **params))

    def load_meta(self):
        if not os.path.exists('meta.json'):
            raise IOError('meta.json not found')

        with open('meta.json', 'r') as fh:
            return json.loads(fh.read())

    def read_meta_field(self, name, allowed=None):
        if not name in self.meta:
            raise RuntimeError(f'{name} field is not in meta.json')
        else:
            val = self.meta[name]

        if allowed is not None and not val in allowed:
            raise RuntimeError(f'invalid field {name} value {val}, only supports {allowed}')
        return val

    def load_handler(self):
        if not os.path.exists('inference/inference.py'):
            raise IOError('inference.py not found')

        if self.processor_type == 'py':
            module = importlib.import_module('inference.inference')
            handler = getattr(module, self.processor_name)()
            handler.initialize(workDir=os.getcwd())
            return handler
        else:
            raise RuntimeError(f'unsupported processor type {self.processor_type}, supportged {self.processorTypes}')
