# Rec-Algorithm-Notes

推荐系统算法学习笔记与代码实现。

本项目用于整理推荐系统相关算法的原理说明、代码实现与示例，便于系统学习与复习。

## 项目结构

```
Rec-Algorithm-Notes/
├── cores/                  # 核心算法实现（可复用模块）
│   ├── layers/             # 通用网络层（Attention、Embedding 等）
│   └── models/             # 模型定义（DIN、YouTube DNN 等）
├── docs/                   # 算法原理与笔记文档
├── examples/               # 各算法独立示例（含 data/ 与 run 脚本）
│   ├── DIN/
│   └── YouTubeDNNRecall/
├── tests/                  # 单元测试
├── requirements.txt        # 基础依赖
└── requirements-dl.txt     # 深度学习依赖（含 torch）
```

## 环境要求

- Python 3.8+

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/suxing99/Rec-Algorithm-Notes.git
cd Rec-Algorithm-Notes

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

```

## 内容规划

- 协同过滤（UserCF / ItemCF）
- 矩阵分解（MF、SVD、ALS）
- 深度学习推荐（DeepFM、Wide & Deep 等）
- 序列推荐与召回排序

## 贡献

欢迎提交 Issue 或 Pull Request。

## License

MIT
