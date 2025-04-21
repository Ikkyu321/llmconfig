import logging
import os

default_round = 50
PARALLEL_SIZE = int(os.environ["PARALLEL_SIZE"])
MODEL_DIR = os.environ["MODEL_DIR"]
WORK_DIR = os.environ["WORK_DIR"]
LLM_INFERENCE_URL = os.environ["LLM_INFERENCE_URL"]
print(f"parallel size is {PARALLEL_SIZE}, model dir is {MODEL_DIR}")
#print(f"work dir is {WORK_DIR}")
print(f"LLM inference url is {LLM_INFERENCE_URL}")

def _setup_logging():
    # 创建一个logger
    logger = logging.getLogger('llmconfig_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志记录的级别

    log_path = f'{MODEL_DIR}/logs/llmconfig.log'
    directory = os.path.dirname(log_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(log_path):
        with open(log_path, "w") as file:
            file.write("logging start >>>>>>>>>>>>>>>>>>>>>")


    file_handler = logging.handlers.TimedRotatingFileHandler(
       log_path, when='midnight', interval=1, backupCount=7)
    file_handler.setLevel(logging.INFO)

    # 创建一个格式器，并将其添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 将处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

_setup_logging()

def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.propagate = False
    return logger