import os
from enum import Enum, auto

# ========== 路径相关 ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
ASR_FLAG_FILE = os.path.join(BASE_DIR, ".asr_download_ok")

# ========== ASR 配置 ==========

# 1. 远程仓库 ID (必须带 iic/ 前缀)
ASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# 2. 本地文件夹名称 (去掉 / 避免路径错误)
_asr_local_name = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
ASR_MODEL_PATH = os.path.join(MODELS_DIR, _asr_local_name)

# ========== TTS 配置 (保持不变) ==========

TTS_MODEL_DIR = os.path.join(MODELS_DIR, "tts_models--zh-CN--baker--tacotron2-DDC-GST")
TTS_MODEL_PATH = os.path.join(TTS_MODEL_DIR, "model_file.pth")
TTS_CONFIG_PATH = os.path.join(TTS_MODEL_DIR, "config_local.json")

TTS_VOCODER_PATH = None
TTS_VOCODER_CONFIG_PATH = None
USE_PYTTXS3_FALLBACK = True

MODEL_PATHS = {
    "asr_model": ASR_MODEL_PATH,
    "tts_model": TTS_MODEL_PATH,
}

# ========== 其他配置 (保持不变) ==========

LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-8RslfTjQR7sUxiwZDNQkOxLmdimQWhj0uaDjAEEAQdkhWWjW")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://metahk.zenymes.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen3-235B-A22B-FP8")

SAMPLE_RATE = 16000
CHUNK_SIZE = 960
MIC_DEVICE_INDEX = 0

class SystemState(Enum):
    INITIALIZING = auto()
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()