# tts.py
import os
import sys
import time
import queue
import tempfile
import subprocess
import multiprocessing
import numpy as np
import re

# 引入音频处理库
import soundfile as sf

# 在导入 torch / TTS 之前关闭 weights_only 限制（适配 torch>=2.6）
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from TTS.api import TTS as CoquiTTS  # 来自 coqui-tts
from config import TTS_MODEL_PATH, TTS_CONFIG_PATH


class TTSEngine(multiprocessing.Process):
    """
    TTS 进程（仅使用本地 Coqui TTS 模型）
    """

    def __init__(self, input_queue, event_queue, audio_device_mock: bool = False):
        super().__init__()
        self.input_queue = input_queue
        self.event_queue = event_queue
        self.mock = audio_device_mock

        self.text_buffer = ""
        self._tts = None  # Coqui TTS 实例

    # ================= 进程主循环 =================

    def run(self):
        print("[TTS] 进程启动（后端：本地 Coqui TTS）")
        # 尝试预初始化一次
        self._ensure_tts()

        if self.mock:
            print("[TTS] mock 模式：只打印文本，不实际播放。")

        while True:
            try:
                data = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (EOFError, KeyboardInterrupt):
                break

            try:
                chunk = data.get("text_chunk", "")
                is_end = data.get("end", False)
            except Exception:
                continue

            if chunk:
                self.text_buffer += chunk

            if is_end:
                text_to_speak = self.text_buffer.strip()
                self.text_buffer = ""

                if text_to_speak:
                    print(f"[TTS] 合成并播放: {text_to_speak}")
                    if not self.mock:
                        self._speak(text_to_speak)
                    else:
                        print("[TTS][mock]", text_to_speak)

                # 通知主进程这一轮 TTS 已结束
                try:
                    self.event_queue.put("TTS_FINISHED")
                except Exception:
                    pass

    # ================= Coqui TTS 初始化 =================

    def _ensure_tts(self):
        """懒加载 Coqui TTS 模型"""
        if self._tts is not None:
            return self._tts

        try:
            print("[TTS] 正在初始化本地 Coqui TTS ...")
            if not (os.path.exists(TTS_MODEL_PATH) and os.path.exists(TTS_CONFIG_PATH)):
                raise FileNotFoundError(f"模型文件未找到: {TTS_MODEL_PATH}")

            self._tts = CoquiTTS(
                model_path=TTS_MODEL_PATH,
                config_path=TTS_CONFIG_PATH,
                progress_bar=False,
                gpu=False,
            )
            print("[TTS] 本地 Coqui TTS 初始化成功。")
        except Exception as e:
            print("[TTS] 初始化本地 Coqui 失败：", repr(e))
            self._tts = None

        return self._tts

    # ================= 播放入口 =================

    def _speak(self, text: str):
        tts = self._ensure_tts()
        if tts is None:
            print("[TTS] 无可用 Coqui 引擎，跳过朗读。")
            return

        try:
            text = self._normalize_text(text)
            if not text:
                print("[TTS] 文本清洗后为空，跳过朗读。")
                return
            self._speak_coqui(tts, text)
        except Exception as e:
            print(f"[TTS] Coqui 合成/播放失败: {e}")

    # ================= Coqui: 合成 -> 切除尾音 -> 播放 =================

    def _speak_coqui(self, tts: CoquiTTS, text: str):
        """
        1. 文本预处理（加句号）
        2. 合成 wav
        3. 轻量后处理：削减尾部静音/杂音，避免削波失真
        4. 播放
        """
        
        # 1. 强制添加标点，帮助模型停止
        if not text.strip().endswith(("。", "！", "？", ".", "!", "?", "…", "~")):
            text += "。"

        # 2. 创建临时文件 (Windows 安全写法：先 close 再用)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.close()
            tmp_wav = f.name

        try:
            # 合成
            tts.tts_to_file(text=text, file_path=tmp_wav)

            # 3. 后处理：去除尾部静音/底噪
            try:
                self._postprocess_wav(tmp_wav)
            except Exception as e:
                print(f"[TTS] 音频后处理失败 (将播放原声): {e}")

            # 4. 播放
            self._play_wav(tmp_wav)

        finally:
            # 5. 清理文件
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass

    def _play_wav(self, path: str):
        if self.mock:
            print(f"[TTS][mock] 模拟播放：{path}")
            return

        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME)
            elif sys.platform == "darwin":
                subprocess.run(["afplay", path], check=False)
            else:
                subprocess.run(["aplay", path], check=False)
        except Exception as e:
            print(f"[TTS] 播放 wav 失败: {e}")

    def _normalize_text(self, text: str) -> str:
        """清洗文本以减少 TTS 词表缺失导致的异常发音。"""
        if not text:
            return ""
        # 去掉英文/拼音/数字，保留中文和常见标点
        text = re.sub(r"[A-Za-z0-9]+", " ", text)
        # 清理不常见符号
        text = re.sub(r"[^\u4e00-\u9fff，。！？、；：,.!?…~\s]", " ", text)
        # 统一空白
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ================= 轻量后处理 =================

    def _postprocess_wav(self, path: str):
        """削减尾部静音/底噪，避免削波失真。"""
        y, sr = sf.read(path, dtype="float32")
        if y.size == 0:
            return

        # 统一用单声道能量计算，但保留原声道数据裁切
        if y.ndim > 1:
            mono = y.mean(axis=1)
        else:
            mono = y

        # 计算帧能量，用于找尾部静音
        win = max(int(sr * 0.02), 1)   # 20ms
        hop = max(int(sr * 0.01), 1)   # 10ms
        n_frames = max((len(mono) - win) // hop + 1, 1)
        rms = np.empty(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop
            frame = mono[start:start + win]
            if len(frame) == 0:
                rms[i] = 0.0
            else:
                rms[i] = np.sqrt(np.mean(frame * frame))

        max_rms = float(np.max(rms))
        if max_rms > 0:
            # 低于最大能量 45dB 认为是静音
            thresh = max_rms * (10 ** (-45.0 / 20.0))
            # 找到最后一个超过阈值的帧
            last_idx = int(np.max(np.where(rms > thresh)[0])) if np.any(rms > thresh) else 0
            tail_keep = int(sr * 0.08)  # 尾部保留 80ms，避免硬切
            end_sample = min((last_idx * hop) + win + tail_keep, len(mono))
            if end_sample > 0:
                y = y[:end_sample]

        # 防止削波失真
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.99:
            y = y * (0.99 / peak)

        # 轻微淡出，减少尾部“啪”声
        fade_len = min(int(sr * 0.025), len(y))  # 25ms
        if fade_len > 1:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            if y.ndim > 1:
                y[-fade_len:, :] *= fade[:, None]
            else:
                y[-fade_len:] *= fade

        sf.write(path, y, sr, subtype="PCM_16")


if __name__ == "__main__":
    # 单文件测试
    q_in = multiprocessing.Queue()
    q_evt = multiprocessing.Queue()
    tts_proc = TTSEngine(q_in, q_evt, audio_device_mock=False)
    tts_proc.start()

    q_in.put({"text_chunk": "你好，这是一段测试音频。", "end": True})
    
    try:
        evt = q_evt.get(timeout=15)
        print("[TEST] 收到事件:", evt)
    except queue.Empty:
        print("[TEST] 超时")

    tts_proc.terminate()
