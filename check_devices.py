#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查系统中可用的音频设备
"""

import pyaudio

def check_audio_devices():
    """检查所有音频设备"""
    print("=== 系统中的音频设备检查 ===")

    p = pyaudio.PyAudio()

    print(f"\n总共发现 {p.get_device_count()} 个音频设备\n")

    print("🎤 输入设备（麦克风）:")
    print("-" * 50)
    input_devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 输入设备
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))
            print(f"设备 {i}: {info['name']}")
            print(f"  输入通道: {info['maxInputChannels']}")
            print(f"  默认采样率: {info['defaultSampleRate']}")
            try:
                host_api = p.get_host_api_info_by_index(info['hostApi'])
                print(f"  主机API: {host_api['name']}")
            except:
                print("  主机API: 未知")
            print()

    if not input_devices:
        print("❌ 未找到任何输入设备！")
    else:
        print(f"✅ 找到 {len(input_devices)} 个输入设备")
        print("\n💡 建议设置:")
        for idx, (device_idx, info) in enumerate(input_devices):
            if "Microphone" in info['name'] or "麦克风" in info['name']:
                print(f"  - MIC_DEVICE_INDEX = {device_idx}  # {info['name']}")
            elif idx == 0:
                print(f"  - MIC_DEVICE_INDEX = {device_idx}  # {info['name']} (第一个设备)")

    print("\n🔊 输出设备（扬声器）:")
    print("-" * 50)
    output_devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 输出设备
        if info['maxOutputChannels'] > 0:
            output_devices.append((i, info))
            print(f"设备 {i}: {info['name']}")
            print(f"  输出通道: {info['maxOutputChannels']}")
            print(f"  默认采样率: {info['defaultSampleRate']}")
            try:
                host_api = p.get_host_api_info_by_index(info['hostApi'])
                print(f"  主机API: {host_api['name']}")
            except:
                print("  主机API: 未知")
            print()

    p.terminate()

    print("\n⚙️  当前config.py设置:")
    try:
        from config import MIC_DEVICE_INDEX
        print(f"MIC_DEVICE_INDEX = {MIC_DEVICE_INDEX}")
    except ImportError:
        print("无法读取config.py")

    print("\n📝 使用说明:")
    print("1. 找到包含'Microphone'或'麦克风'的设备")
    print("2. 记录该设备的索引号")
    print("3. 在config.py中设置 MIC_DEVICE_INDEX = 该索引号")
    print("4. 重新运行程序")

if __name__ == "__main__":
    check_audio_devices()
