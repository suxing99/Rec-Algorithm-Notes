# Rec-Algorithm-Notes

推荐系统算法学习笔记与代码实现。

本项目用于整理推荐系统相关算法的原理说明、代码实现与示例，便于系统学习与复习。

## 项目结构

```
Rec-Algorithm-Notes/
├── cores/      # 核心算法实现
├── docs/       # 算法原理与笔记文档
├── examples/   # 使用示例
└── tests/      # 单元测试
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

# 安装依赖（如有 requirements.txt）
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
