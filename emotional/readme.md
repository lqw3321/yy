## Environments

- Python 3.8
- Keras & TensorFlow 2


&nbsp;

## Structure

```
├── models/                // 模型实现
│   ├── base.py          // 所有模型的基类    
│   └── ml.py              // SVM & MLP
├── extract_feats/         // 特征提取
│   ├── librosa.py         // librosa 提取特征
├── utils/
│   ├── files.py           // 用于整理数据集（分类、批量重命名）
│   ├── opts.py            // 使用 argparse 从命令行读入参数
│   └── plot.py            // 绘图（雷达图、频谱图、波形图）
├── config/                // 配置参数（.yaml）
├── features/              // 存储提取好的特征
├── checkpoints/           // 存储训练好的模型权重
├── train.py               // 训练模型
├── predict.py             // 用训练好的模型预测指定音频的情感
└── preprocess.py          // 数据预处理（提取数据集中音频的特征并保存）


安装依赖：

```python
pip install -r requirements.txt