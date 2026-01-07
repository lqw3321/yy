#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声纹注册工具 v4.0
智能声纹注册系统
支持音频质量监控、实时反馈、多重验证
"""

import os
import sys
import time
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple, Dict
import threading
import queue

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from speaker import ECAPATDNNRecognizer
from config import SAMPLE_RATE, MIN_ENROLLMENT_SAMPLES, MIC_DEVICE_INDEX, SPEAKER_SIMILARITY_THRESHOLD
from enhancement import AudioEnhancer


class AudioQualityAnalyzer:
    """音频质量分析器"""

    def __init__(self):
        self.quality_thresholds = {
            'min_rms': 0.08,      # 最小有效RMS
            'max_rms': 0.7,       # 最大RMS（避免过载）
            'min_length': 1.5,    # 最短音频长度
            'max_silence': 0.2,   # 最大静音比例
        }

    def analyze_audio(self, audio_data: bytes) -> Dict:
        """分析音频质量"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            length = len(audio_np) / SAMPLE_RATE

            # 基本统计
            rms = np.sqrt(np.mean(audio_np**2))
            peak = np.max(np.abs(audio_np))
            silence_frames = np.sum(np.abs(audio_np) < 0.01)
            silence_ratio = silence_frames / len(audio_np)

            # 质量评分
            quality_score = self._calculate_quality_score(rms, length, silence_ratio)

            return {
                'rms': rms,
                'peak': peak,
                'length': length,
                'silence_ratio': silence_ratio,
                'quality_score': quality_score,
                'is_acceptable': quality_score >= 0.6,
                'feedback': self._get_feedback(rms, length, silence_ratio, quality_score)
            }
        except Exception as e:
            return {
                'error': str(e),
                'is_acceptable': False,
                'feedback': f"音频分析失败: {e}"
            }

    def _calculate_quality_score(self, rms: float, length: float, silence_ratio: float) -> float:
        """计算质量分数 (0-1)"""
        score = 0.0

        # RMS评分 (40%)
        if rms < self.quality_thresholds['min_rms']:
            rms_score = rms / self.quality_thresholds['min_rms'] * 0.5
        elif rms > self.quality_thresholds['max_rms']:
            rms_score = 0.5
        else:
            # 最佳范围 0.15-0.4
            if 0.15 <= rms <= 0.4:
                rms_score = 1.0
            else:
                rms_score = 0.8

        # 长度评分 (30%)
        if length < self.quality_thresholds['min_length']:
            length_score = length / self.quality_thresholds['min_length']
        else:
            length_score = min(length / 3.0, 1.0)

        # 静音评分 (30%)
        silence_score = 1.0 - min(silence_ratio / self.quality_thresholds['max_silence'], 1.0)

        score = 0.4 * rms_score + 0.3 * length_score + 0.3 * silence_score
        return min(max(score, 0.0), 1.0)

    def _get_feedback(self, rms: float, length: float, silence_ratio: float, score: float) -> str:
        """生成质量反馈"""
        feedback = []

        if score >= 0.8:
            feedback.append("✅ 音频质量优秀！")
        elif score >= 0.6:
            feedback.append("👍 音频质量良好")
        else:
            feedback.append("⚠️ 音频质量需要改进")

        # 具体建议
        if rms < self.quality_thresholds['min_rms']:
            feedback.append(".3f"        elif rms > self.quality_thresholds['max_rms']:
            feedback.append(".3f"        if length < self.quality_thresholds['min_length']:
            feedback.append(".1f"        if silence_ratio > self.quality_thresholds['max_silence']:
            feedback.append(".1%")

        return " | ".join(feedback)


class SmartSpeakerRegistrationTool:
    """智能声纹注册工具 v4.0"""

    def __init__(self):
        self.recognizer = ECAPATDNNRecognizer()
        self.quality_analyzer = AudioQualityAnalyzer()
        self.enhancer = AudioEnhancer()

        # 注册会话状态
        self.session_stats = {
            'user_id': None,
            'attempts': 0,
            'successful_samples': 0,
            'quality_scores': [],
            'best_quality': 0.0
        }

        print("🎤 智能声纹注册工具 v4.0 初始化完成")
        print("✨ 支持实时质量监控和智能引导")

    def record_audio_with_quality_check(self, duration: float = 3.0, show_feedback: bool = True) -> Tuple[Optional[bytes], Dict]:
        """智能录音函数，包含质量分析"""
        try:
            import pyaudio

            chunk = 1024
            format = pyaudio.paInt16
            channels = 1
            rate = SAMPLE_RATE

            p = pyaudio.PyAudio()

            # 验证设备
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, MIC_DEVICE_INDEX)
                if device_info.get('maxInputChannels') <= 0:
                    return None, {'error': f'设备 {MIC_DEVICE_INDEX} 没有输入通道'}
                if show_feedback:
                    print(f"🎤 使用设备: {device_info['name']} (索引: {MIC_DEVICE_INDEX})")
            except Exception as e:
                return None, {'error': f'无法访问设备 {MIC_DEVICE_INDEX}: {e}'}

            # 打开音频流
            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=chunk
            )

            # 显示使用的设备
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, MIC_DEVICE_INDEX)
                print(f"🎤 使用设备: {device_info['name']} (索引: {MIC_DEVICE_INDEX})")
            except Exception as e:
                print(f"⚠️ 无法获取设备信息: {e}")

            stream = p.open(
                format=format,
                channels=channels,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=MIC_DEVICE_INDEX,
                frames_per_buffer=chunk
            )

            print(f"🎙️ 开始录音 {duration} 秒...")
            print("请清晰地说出句子，保持自然语速")
            print("💡 提示：保持15-30cm距离，音量适中")

            frames = []

            # 显示进度条
            total_chunks = int(duration * SAMPLE_RATE / chunk)
            for i in range(total_chunks):
                data = stream.read(chunk)
                frames.append(data)

                # 每0.5秒显示一次进度
                if i % int(0.5 * SAMPLE_RATE / chunk) == 0:
                    progress = (i + 1) / total_chunks
                    bar_length = 20
                    filled = int(bar_length * progress)
                    bar = "█" * filled + "░" * (bar_length - filled)
                    print(f"\r⏱️ 录音进度: [{bar}] {progress:.1%}", end="", flush=True)

            print(" ✅")  # 完成进度条

            stream.stop_stream()
            stream.close()
            p.terminate()

            # 合并音频数据
            audio_data = b''.join(frames)

            # 分析音频质量
            quality_info = self.quality_analyzer.analyze_audio(audio_data)

            return audio_data, quality_info

        except Exception as e:
            print(f"❌ 录音失败: {e}")
            return None

    def _analyze_audio(self, audio_data: bytes) -> dict:
        """分析音频质量"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            length = len(audio_np) / SAMPLE_RATE
            rms = np.sqrt(np.mean(audio_np**2))

            if length < 1.0:
                return {'valid': False, 'reason': '音频太短'}
            if rms < 0.01:
                return {'valid': False, 'reason': '音频信号太弱', 'rms': rms}

            return {
                'valid': True,
                'length': length,
                'rms': rms,
                'quality': '良好' if rms > 0.1 else '一般'
            }
        except Exception as e:
            return {'valid': False, 'reason': f'分析错误: {e}'}

    def register_user_interactive(self):
        """交互式注册用户"""
        print("\n" + "="*50)
        print("🎤 声纹注册系统 v3.0")
        print("="*50)

        # 输入用户名
        while True:
            user_id = input("\n请输入用户名（字母、数字、下划线）：").strip()
            if not user_id:
                print("❌ 用户名不能为空")
                continue

            if not all(c.isalnum() or c == '_' for c in user_id):
                print("❌ 用户名只能包含字母、数字和下划线")
                continue

            break

        # 检查用户是否已存在
        existing_users = self.recognizer.get_user_list()
        if user_id in existing_users:
            count = self.recognizer.get_user_count(user_id)
            print(f"ℹ️  用户 '{user_id}' 已存在，当前有 {count} 个样本")

            choice = input("是否继续添加新样本？(y/n): ").strip().lower()
            if choice != 'y':
                print("注册取消")
                return

        # 开始注册流程
        print(f"\n🎯 开始为用户 '{user_id}' 注册声纹")
        print(f"需要录制 {MIN_ENROLLMENT_SAMPLES} 个语音样本")

        sample_sentences = [
            "今天天气真不错",
            "我喜欢听音乐",
            "谢谢你的帮助",
            "人工智能发展很快",
            "语音识别技术很有趣"
        ]

        collected_samples = 0
        while collected_samples < MIN_ENROLLMENT_SAMPLES:
            print(f"\n📝 录制第 {collected_samples + 1}/{MIN_ENROLLMENT_SAMPLES} 个样本")

            # 显示建议句子
            if collected_samples < len(sample_sentences):
                print(f"建议句子：'{sample_sentences[collected_samples]}'")
            else:
                print("请说任意一句自然的话")

            # 录音
            audio_data = self._record_audio(duration=3.0)
            if audio_data is None:
                continue

            # 分析音频质量
            analysis = self._analyze_audio(audio_data)
            if not analysis['valid']:
                print(f"❌ {analysis['reason']}")
                print("请重新录制")
                continue

            print(".2f")
            print(f"音频质量: {analysis['quality']}")

            # 注册样本
            success = self.recognizer.enroll_user(user_id, audio_data)
            if success:
                collected_samples += 1
                print(f"✅ 第 {collected_samples}/{MIN_ENROLLMENT_SAMPLES} 个样本注册成功！")
            else:
                print("❌ 注册失败，请重试")

            if collected_samples < MIN_ENROLLMENT_SAMPLES:
                input("\n按回车键继续录制下一个样本...")

        # 注册完成
        print(f"\n🎉 注册完成！")
        final_count = self.recognizer.get_user_count(user_id)
        print(f"👤 用户: {user_id}")
        print(f"📊 注册样本数: {final_count}")

        # 显示所有用户
        all_users = self.recognizer.get_user_list()
        print(f"\n👥 已注册用户 ({len(all_users)} 个):")
        for user in all_users:
            count = self.recognizer.get_user_count(user)
            status = "✅" if count >= MIN_ENROLLMENT_SAMPLES else "⚠️"
            print(f"  {status} {user}: {count} 个样本")

    def run(self):
        """运行注册工具"""
        self.register_user_interactive()


def main():
    """主函数"""
    try:
        tool = SpeakerRegistrationTool()
        tool.run()
    except KeyboardInterrupt:
        print("\n👋 注册已取消")
    except Exception as e:
        print(f"❌ 程序出错: {e}")


if __name__ == "__main__":
    main()

    def record_audio(self, duration: float = 3.0) -> Optional[bytes]:
        """从麦克风录制音频"""
        try:
            import pyaudio
        except ImportError:
            print("错误：未安装pyaudio，无法录音")
            print("请运行：pip install pyaudio")
            return None

        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        sample_rate = SAMPLE_RATE

        p = pyaudio.PyAudio()

        try:
            # 检查是否有可用的输入设备
            info = p.get_host_api_info_by_index(0)
            device_count = info.get('deviceCount')

            # 使用与main.py相同的设备索引
            input_device = MIC_DEVICE_INDEX

            # 验证设备有效性
            try:
                device_info = p.get_device_info_by_host_api_device_index(0, input_device)
                print(f"🎤 使用设备: {device_info['name']} (索引: {input_device})")

                if device_info.get('maxInputChannels') <= 0:
                    print(f"错误：设备 {input_device} 没有输入通道")
                    print("请检查config.py中的MIC_DEVICE_INDEX设置")
                    return None
            except Exception as e:
                print(f"错误：无法获取设备 {input_device} 信息: {e}")
                print("请检查config.py中的MIC_DEVICE_INDEX设置")
                return None

            stream = p.open(format=format,
                          channels=channels,
                          rate=sample_rate,
                          input=True,
                          input_device_index=input_device,
                          frames_per_buffer=chunk)

            print(f"🎤 开始录音 {duration} 秒...")
            print("请清晰地说出句子，保持自然语速")

            frames = []

            # 显示倒计时
            for i in range(int(duration * 10)):
                remaining = duration - (i * 0.1)
                print(f"\r⏱️  剩余时间: {remaining:.1f}秒", end="", flush=True)
                time.sleep(0.1)

                # 每0.1秒读一次数据
                if i % 1 == 0:  # 每10次循环（1秒）收集数据
                    data = stream.read(chunk)
                    frames.append(data)

            print("\n✅ 录音完成！")

            stream.stop_stream()
            stream.close()

            return b''.join(frames)

        except Exception as e:
            print(f"录音失败: {e}")
            return None
        finally:
            p.terminate()

    def generate_test_audio(self, frequency: float = 440.0, duration: float = 3.0) -> bytes:
        """生成测试音频（用于演示）"""
        sample_rate = SAMPLE_RATE
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # 生成简单的正弦波
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)

        # 添加一些谐波使声音更自然
        audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)

        # 添加轻微噪声
        audio += 0.05 * np.random.normal(0, 1, len(audio))

        # 转换为16位PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def register_user_interactive(self):
        """交互式注册用户"""
        print("\n" + "="*50)
        print("🎤 声纹注册系统 - 交互式注册")
        print("="*50)

        # 输入用户名
        while True:
            user_id = input("\n请输入用户名（字母、数字、下划线）：").strip()
            if not user_id:
                print("❌ 用户名不能为空")
                continue

            # 检查用户名格式
            if not all(c.isalnum() or c == '_' for c in user_id):
                print("❌ 用户名只能包含字母、数字和下划线")
                continue

            break

        # 检查用户是否已存在
        existing_users = self.recognizer.get_user_list()
        if user_id in existing_users:
            count = self.recognizer.get_user_count(user_id)
            print(f"ℹ️  用户 '{user_id}' 已存在，当前有 {count} 个样本")

            choice = input("是否继续添加新样本？(y/n): ").strip().lower()
            if choice != 'y':
                print("注册取消")
                return

        # 开始注册流程
        print(f"\n🎯 开始为用户 '{user_id}' 注册声纹")
        print("需要录制多个语音样本以提高识别准确率")
        print("建议：每个样本说不同的句子，保持自然语速")

        required_samples = MIN_ENROLLMENT_SAMPLES
        collected_samples = 0
        sample_sentences = [
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

        while collected_samples < required_samples:
            print(f"\n" + "-"*30)
            print(f"📝 录制第 {collected_samples + 1}/{required_samples} 个样本")

            # 建议句子
            if collected_samples < len(sample_sentences):
                print(f"建议句子：'{sample_sentences[collected_samples]}'")
            else:
                print("请说任意一句自然的话")

            # 询问是否使用真实录音还是测试音频
            use_real_audio = input("使用真实录音？(y=真实录音, n=测试音频): ").strip().lower()

            if use_real_audio == 'y':
                audio_data = self.record_audio(duration=3.0)
                if audio_data is None:
                    print("❌ 录音失败，跳过此样本")
                    continue
            else:
                # 使用测试音频
                frequency = 440 + (collected_samples * 50)  # 不同的频率模拟不同语音
                audio_data = self.generate_test_audio(frequency=frequency, duration=3.0)
                print("✅ 已生成测试音频")

            # 显示识别内容
            if use_real_audio == 'y':
                print("🔍 正在分析您刚才说的内容...")
                recognized_text = self.recognize_audio_content(audio_data)
                print(f"🎙️ 识别结果: 【{recognized_text}】")

                # 询问是否确认
                confirm = input("内容是否正确？(y=确认, n=重新录制, s=跳过确认): ").strip().lower()
                if confirm == 'n':
                    print("🔄 重新录制此样本...")
                    continue  # 重新录制，不增加collected_samples
                elif confirm == 's':
                    print("⏭️ 跳过确认，继续注册...")
                # 如果是'y'或其他，直接继续

            # 注册样本
            success = self.recognizer.enroll_user(user_id, audio_data)

            if success:
                collected_samples += 1
                print(f"✅ 第 {collected_samples}/{required_samples} 个样本注册成功！")
            else:
                print(f"❌ 第 {collected_samples + 1} 个样本注册失败")

            # 如果不是最后一个样本，稍作停顿
            if collected_samples < required_samples:
                input("\n按回车键继续录制下一个样本...")

        # 注册完成
        print(f"\n" + "="*50)
        if collected_samples > 0:
            final_count = self.recognizer.get_user_count(user_id)
            print(f"🎉 注册完成！")
            print(f"👤 用户名: {user_id}")
            print(f"📊 注册样本数: {final_count}")

            if final_count >= 3:
                print("✅ 注册成功！现在可以进行声纹识别了")
            else:
                print("⚠️  建议再注册一些样本以提高准确率")
        else:
            print("❌ 注册失败，没有成功注册任何样本")

        # 显示所有用户
        self.show_registered_users()

    def register_user_batch(self, user_id: str, audio_files: List[str]):
        """批量注册用户从音频文件"""
        print(f"\n📁 批量注册用户 '{user_id}'")
        print(f"从 {len(audio_files)} 个音频文件注册")

        success_count = 0

        for i, audio_file in enumerate(audio_files, 1):
            if not os.path.exists(audio_file):
                print(f"❌ 文件不存在: {audio_file}")
                continue

            try:
                # 读取音频文件
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_file, dtype='int16')

                # 如果是立体声，转为单声道
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                # 转换为bytes
                audio_bytes = audio_data.tobytes()

                # 注册
                success = self.recognizer.enroll_user(user_id, audio_bytes)

                if success:
                    success_count += 1
                    print(f"✅ 文件 {i}/{len(audio_files)}: {os.path.basename(audio_file)} 注册成功")
                else:
                    print(f"❌ 文件 {i}/{len(audio_files)}: {os.path.basename(audio_file)} 注册失败")

            except Exception as e:
                print(f"❌ 处理文件 {audio_file} 时出错: {e}")

        print(f"\n📊 批量注册完成: {success_count}/{len(audio_files)} 个文件成功")
        final_count = self.recognizer.get_user_count(user_id)
        print(f"👤 用户 '{user_id}' 共有 {final_count} 个注册样本")

    def show_registered_users(self):
        """显示所有注册用户"""
        users = self.recognizer.get_user_list()

        if not users:
            print("\n📭 当前没有注册用户")
            return

        print(f"\n👥 已注册用户 ({len(users)} 个):")
        print("-" * 30)

        for user in users:
            count = self.recognizer.get_user_count(user)
            status = "✅" if count >= 3 else "⚠️ "
            print(f"  {status} {user}: {count} 个样本")

    def test_recognition(self):
        """测试声纹识别功能"""
        print("\n🧪 声纹识别测试")
        print("-" * 30)

        users = self.recognizer.get_user_list()
        if not users:
            print("❌ 没有注册用户，无法测试识别")
            return

        print("已注册用户:", ", ".join(users))

        # 使用测试音频进行识别
        test_frequency = 440.0  # 使用第一个用户的频率
        test_audio = self.generate_test_audio(frequency=test_frequency, duration=2.0)

        print("🎤 正在识别测试音频...")
        result = self.recognizer.identify(test_audio)

        if result != "unknown":
            print(f"✅ 识别结果: {result}")
        else:
            print("❌ 识别失败，未匹配到已知用户")

    def manage_users(self):
        """用户管理"""
        while True:
            print("\n👥 用户管理")
            print("-" * 20)
            print("1. 查看所有用户")
            print("2. 删除用户")
            print("3. 清空所有用户")
            print("4. 返回主菜单")

            choice = input("\n请选择操作 (1-4): ").strip()

            if choice == "1":
                self.show_registered_users()

            elif choice == "2":
                users = self.recognizer.get_user_list()
                if not users:
                    print("❌ 没有注册用户")
                    continue

                print("已注册用户:")
                for i, user in enumerate(users, 1):
                    count = self.recognizer.get_user_count(user)
                    print(f"  {i}. {user} ({count} 个样本)")

                try:
                    user_choice = input("请输入要删除的用户编号: ").strip()
                    user_index = int(user_choice) - 1

                    if 0 <= user_index < len(users):
                        user_to_delete = users[user_index]
                        confirm = input(f"确定要删除用户 '{user_to_delete}' 吗？(y/n): ").strip().lower()

                        if confirm == 'y':
                            success = self.recognizer.remove_user(user_to_delete)
                            if success:
                                print(f"✅ 用户 '{user_to_delete}' 已删除")
                            else:
                                print(f"❌ 删除用户 '{user_to_delete}' 失败")
                        else:
                            print("已取消删除")
                    else:
                        print("❌ 无效的用户编号")

                except ValueError:
                    print("❌ 请输入有效的数字")

            elif choice == "3":
                confirm = input("⚠️  确定要清空所有用户数据吗？此操作不可恢复！(y/n): ").strip().lower()
                if confirm == 'y':
                    success = self.recognizer.clear_database()
                    if success:
                        print("✅ 已清空所有用户数据")
                    else:
                        print("❌ 清空失败")
                else:
                    print("已取消清空操作")

            elif choice == "4":
                break

            else:
                print("❌ 无效选择，请重新输入")

    def run(self):
        """主运行函数"""
        print("🎤 声纹注册工具 v1.0")
        print("基于ECAPA-TDNN的声纹识别系统")

        while True:
            print("\n" + "="*50)
            print("主菜单")
            print("="*50)
            print("1. 📝 交互式注册用户")
            print("2. 📁 批量注册（从音频文件）")
            print("3. 👥 查看注册用户")
            print("4. 🧪 测试声纹识别")
            print("5. ⚙️  用户管理")
            print("6. 🚪 退出")

            choice = input("\n请选择功能 (1-6): ").strip()

            if choice == "1":
                self.register_user_interactive()

            elif choice == "2":
                user_id = input("请输入用户名: ").strip()
                if not user_id:
                    print("❌ 用户名不能为空")
                    continue

                files_input = input("请输入音频文件路径（用逗号分隔）: ").strip()
                if not files_input:
                    print("❌ 文件路径不能为空")
                    continue

                audio_files = [f.strip() for f in files_input.split(",") if f.strip()]
                if not audio_files:
                    print("❌ 没有有效的文件路径")
                    continue

                self.register_user_batch(user_id, audio_files)

            elif choice == "3":
                self.show_registered_users()

            elif choice == "4":
                self.test_recognition()

            elif choice == "5":
                self.manage_users()

            elif choice == "6":
                print("👋 感谢使用声纹注册工具！")
                break

            else:
                print("❌ 无效选择，请重新输入")

            # 暂停一下，让用户看到结果
            input("\n按回车键继续...")


def main():
    """主函数"""
    try:
        tool = SpeakerRegistrationTool()
        tool.run()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


