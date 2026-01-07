#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成注册管理器
在主系统运行时提供声纹注册功能
"""

import time
import threading
from typing import Optional, Any

from config import MIN_ENROLLMENT_SAMPLES


class IntegratedRegistrationManager:
    """集成注册管理器"""

    def __init__(self, audio_device, speaker_recognizer, audio_enhancer,
                 asr_queue, text_queue):
        self.audio_device = audio_device
        self.speaker_recognizer = speaker_recognizer
        self.audio_enhancer = audio_enhancer
        self.asr_queue = asr_queue
        self.text_queue = text_queue

        # 注册状态
        self.is_registering = False
        self.registration_buffer = bytearray()
        self.collected_samples = 0
        self.current_user_id = ""

        # 示例句子
        self.sample_sentences = [
            "今天天气真不错",
            "我喜欢听音乐",
            "谢谢你的帮助",
            "这是一个测试句子",
            "语音识别技术很有趣",
            "人工智能发展很快",
            "请问现在几点了",
            "我想听一首歌",
            "这个功能很实用",
            "声纹识别真神奇"
        ]

    def run_registration(self):
        """运行集成注册流程"""
        print("\n" + "="*60)
        print("🎤 声纹注册模式 - 集成到主系统")
        print("="*60)
        print("✓ 使用主系统的麦克风设备")
        print("✓ 使用主系统的ASR引擎")
        print("✓ 使用主系统的音频增强")
        print("✓ 实时显示识别结果")
        print("="*60)

        # 输入用户名
        while True:
            try:
                user_id = input("\n请输入用户名（字母、数字、下划线）：").strip()
                if not user_id:
                    print("❌ 用户名不能为空")
                    continue

                # 检查用户名格式
                if not all(c.isalnum() or c == '_' for c in user_id):
                    print("❌ 用户名只能包含字母、数字和下划线")
                    continue

                break
            except KeyboardInterrupt:
                print("\n❌ 注册取消")
                return

        self.current_user_id = user_id

        # 检查用户是否已存在
        existing_users = self.speaker_recognizer.get_user_list()
        if user_id in existing_users:
            count = self.speaker_recognizer.get_user_count(user_id)
            print(f"ℹ️  用户 '{user_id}' 已存在，当前有 {count} 个样本")

            choice = input("是否继续添加新样本？(y/n): ").strip().lower()
            if choice != 'y':
                print("注册取消")
                return

        # 开始注册流程
        print(f"\n🎯 开始为用户 '{user_id}' 注册声纹")
        print(f"需要录制 {MIN_ENROLLMENT_SAMPLES} 个语音样本")
        print("建议：每个样本说不同的句子，保持自然语速")
        print("\n注册指令：")
        print("- 按 [r] 开始录制")
        print("- 按 [s] 停止录制")
        print("- 按 [q] 退出注册")

        self._run_registration_loop()

    def _run_registration_loop(self):
        """注册主循环"""
        print(f"\n准备录制第 {self.collected_samples + 1}/{MIN_ENROLLMENT_SAMPLES} 个样本")
        print("按 [r] 开始录制，[s] 停止，[q] 退出")

        while self.collected_samples < MIN_ENROLLMENT_SAMPLES:
            try:
                cmd = input("命令: ").strip().lower()

                if cmd == 'q':
                    print("❌ 注册取消")
                    return
                elif cmd == 'r':
                    self._start_recording()
                elif cmd == 's':
                    self._stop_recording()
                else:
                    print("无效命令。按 [r] 开始，[s] 停止，[q] 退出")

            except KeyboardInterrupt:
                print("\n❌ 注册取消")
                return

        # 注册完成
        print(f"\n🎉 注册完成！")
        final_count = self.speaker_recognizer.get_user_count(self.current_user_id)
        print(f"👤 用户: {self.current_user_id}")
        print(f"📊 注册样本数: {final_count}")

    def _start_recording(self):
        """开始录制"""
        if self.is_registering:
            print("⚠️ 已经在录制中")
            return

        if not self.audio_device:
            print("❌ 音频设备未初始化")
            return

        print(f"\n🎙️ 开始录制第 {self.collected_samples + 1} 个样本...")
        print("请清晰地说出句子，保持自然语速")
        print("说完后按 [s] 停止录制")

        self.is_registering = True
        self.registration_buffer.clear()

        # 开始实际录音
        print("⏱️ 录制中...")

        # 这里可以设置一个录制线程或者循环采集音频
        # 暂时使用简单的循环
        recording_start_time = time.time()
        max_recording_time = 10  # 最大录制10秒

        try:
            while self.is_registering and (time.time() - recording_start_time) < max_recording_time:
                if self.audio_device:
                    # 从音频设备读取数据
                    audio_chunk = self.audio_device.read_chunk()
                    if audio_chunk:
                        self.registration_buffer.extend(audio_chunk)
                time.sleep(0.01)  # 小延迟避免CPU占用过高

        except KeyboardInterrupt:
            pass

        print("✅ 录制完成！")
        self._stop_recording()

    def _stop_recording(self):
        """停止录制并处理样本"""
        if not self.is_registering:
            print("⚠️ 当前没有在录制")
            return

        self.is_registering = False

        if len(self.registration_buffer) == 0:
            print("❌ 没有录制到音频数据")
            return

        print("🔍 正在处理音频样本...")

        try:
            audio_data = bytes(self.registration_buffer)

            # 显示ASR识别结果（暂时使用简化识别）
            recognized_text = self._simple_asr_recognize(audio_data)
            print(f"🎙️ 识别结果: 【{recognized_text}】")

            # 询问用户是否确认
            confirm = input("内容是否正确？(y=确认注册, n=重新录制, s=跳过确认直接注册): ").strip().lower()

            if confirm == 'n':
                print("🔄 重新录制此样本...")
                return
            elif confirm == 's':
                print("⏭️ 跳过确认，直接注册...")
            # 如果是'y'或其他，继续注册

            # 使用声纹识别器注册样本
            success = self.speaker_recognizer.enroll_user(self.current_user_id, audio_data)

            if success:
                self.collected_samples += 1
                print(f"✅ 第 {self.collected_samples}/{MIN_ENROLLMENT_SAMPLES} 个样本注册成功！")

                if self.collected_samples < MIN_ENROLLMENT_SAMPLES:
                    print(f"\n准备录制第 {self.collected_samples + 1} 个样本")
                    print("按 [r] 开始录制，[s] 停止，[q] 退出")
            else:
                print("❌ 样本注册失败，请重试")

        except Exception as e:
            print(f"❌ 处理样本时出错: {e}")

    def _simple_asr_recognize(self, audio_data: bytes) -> str:
        """简化的ASR识别（用于测试）"""
        try:
            # 检查音频质量
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_length = len(audio_np) / 16000  # 假设16kHz采样率
            rms = np.sqrt(np.mean(audio_np**2))

            if audio_length < 1.0:
                return "音频太短（不足1秒）"
            if rms < 0.01:
                return "音频信号太弱（可能是静音）"

            return f"音频质量正常 (长度:{audio_length:.1f}秒, RMS:{rms:.3f}) - 暂时跳过ASR识别"

        except Exception as e:
            return f"音频检查错误: {e}"

    def _get_asr_result(self, audio_data: bytes) -> str:
        """获取ASR识别结果（保留原有实现）"""
        return self._simple_asr_recognize(audio_data)  # 暂时使用简化版本
