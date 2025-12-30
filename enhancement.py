import numpy as np


class AudioEnhancer:
    """
    语音增强模块
    负责对原始音频进行降噪、自动增益控制(AGC)或静音消除(VAD)。
    """

    def __init__(self):
        print("[Enhancement] 语音增强模块加载完毕 (当前模式: 直通)")

    def process(self, audio_data: bytes) -> bytes:
        """
        处理音频数据
        :param audio_data: 原始 PCM 音频流 (bytes)
        :return: 处理后的音频流 (bytes)
        """
        # --- 接入真实算法建议 ---
        # 1. 如果要去除静音: 使用 webrtcvad
        # 2. 如果要降噪: 使用 noisereduce 库
        # 3. 简单实现: 只有当音量超过某个阈值时才通过 (Noise Gate)

        # 目前：不做任何处理，直接返回 (Pass-through)
        return audio_data