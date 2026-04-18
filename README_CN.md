<h1 align="center">OmniWriter: 基于LangGraph + DeepAgents的生产级多智能体文章生成系统</h1>

<p align="center">
   Languages: 
   简体中文
   <a href="./docs/README.md">English.</a>
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

## 📸 核心特性

- 🧠 **多智能体协作**：基于 LangGraph 编排工作流，DeepAgents 负责后期处理，分工明确且可扩展
- 🎯 **幻觉抑制**：结构化抽取检索词、精准提示约束、Rerank 检索过滤、被动 + 主动检索多层保障
- 🔍 **高效检索**：关键词检索 (ES) + 向量检索 (ChromaDB) 混合模式，精准三元组检索词 + 元数据筛选优化效果
- 📦 **上下文管理**：分阶段压缩、虚拟文件系统持久化、DeepAgents 自动消息整理，避免上下文溢出
- ⚡ **容错机制**：重试策略、格式错误自动重试、异常日志专属处理节点，提升系统稳定性
- 🚀 **并行化处理**：子任务并行检索、章节并行撰写、图片并行生成，大幅提升生成效率
- 📋 **全流程可控**：大纲中断审核、多轮迭代优化、用户自定义修改，满足个性化需求

## 🛠️ 技术栈

|    类别    |                     技术 / 框架                     |
| :--------: | :-------------------------------------------------: |
|  编程语言  |                    Python 3.12+                     |
| 智能体框架 | LangGraph（工作流编排）、DeepAgents（多智能体协作） |
|   数据库   |  ChromaDB（向量数据库）、Elasticsearch（全文检索）  |
|  模型监控  |                      LangSmith                      |
|  网页检索  |                     Travily API                     |
|  网络请求  |                  requests、aiohttp                  |
|  依赖管理  |                         uv                          |
|  日志系统  |            Python logging（自定义配置）             |

## 📊 核心工作流

OmniWriter 将文章生成拆解为 5 个核心阶段，全流程自动化且可迭代：

### 阶段 1：任务拆解与全景检索

- 主题分解：将用户输入主题拆分为互不重叠的子任务
- 并行搜索：协程机制多智能体全网拉取原始资料
- 向量库构建：Chunk 分块 + 向量化，存入 ChromaDB（内存临时库）+ ES，打分子任务元数据

### 阶段 2：素材提炼与骨架规划

- 子任务结果浓缩：结构化抽取（总结 + 逻辑 + 压缩检索词列表）
- 全局大纲生成：基于浓缩素材规划文章骨架，支持用户审核 / 修改
- 中断节点：大纲保存，用户确认后继续流程

### 阶段 3：并行起草与全局缝合

- 并行章节撰写：多写作智能体同时撰写不同章节，写作前检索对应素材防幻觉
- 章节拼接：专属节点拼接完整文章，持久化到用户虚拟文件系统

### 阶段 4：后期处理（DeepAgents）

主智能体调度子智能体完成全维度优化：

- 润色智能体：内容优化 + 图片 / 代码描述补充
- 审核智能体：100 分制审核，低于 80 分触发重润色
- 画图智能体：并行生成图片，保存到用户目录
- 排版智能体：替换占位符 + 按需求排版（默认 Markdown）

### 阶段 5：迭代优化

用户反馈修改意见，路由智能体判断并定向执行对应节点（无需全流程重跑）

## 🎯 关键问题解决方案

|  核心问题  |                           解决方案                           |
| :--------: | :----------------------------------------------------------: |
| 大模型幻觉 | 压缩检索词列表 + 精准提示约束 + Rerank 检索过滤 + 被动 / 主动检索补充 |
| 检索效果差 | 精准三元组检索词 + 元数据筛选 + ES+ChromaDB 混合检索 + 原始素材回退 |
| 上下文溢出 |  分阶段压缩 + 虚拟文件系统持久化 + DeepAgents 自动消息整理   |
| 系统容错性 | 可恢复错误重试 + 格式错误 LLM 重试 + 异常专属处理节点 + 完整日志记录 |

## 📂 项目结构

```
OmniWriter/
├── .env                               # 环境变量配置文件（API Key等）
├── .gitignore                         # Git忽略文件配置
├── README.md                          # 
├── uv.lock                            # Git忽略文件配置
├── pyproject.toml                     # Python项目依赖与配置
├── main.py                            # 项目入口文件
├── data/
│   ├── log/                           # 日志
│   ├── chromadb/                      # 向量数据库
│   └── {user_id}/                     # 用户文件系统
│       ├── images/                    # 生成的图片
│       ├── skills/                    # 用户自定义的skills
│       ├── supplementary_material     # 用户提供的素材
│       ├── article.txt                # 生成的文章
│       └── outline.txt                # 生成的大纲
│       
├── src/
│   ├── core/                          # 核心客户端模块（向量库、搜索引擎、大模型）
│   │   ├── __init__.py
│   │   ├── chroma_client.py           # Chroma向量数据库客户端
│   │   ├── es_client.py               # Elasticsearch搜索引擎客户端
│   │   └── model_client.py            # 大模型API客户端封装
│   │
│   └── pipeline/                      # 多智能体工作流核心模块
│       ├── agents/                    # 所有智能体实现
│       │   ├── deepagent/             # deepagents智能体子模块
│       │   │   ├── __init__.py
│       │   │   ├── controller_agent.py    # 流程控制智能体
│       │   │   ├── draw_images_agent.py   # 图片生成智能体
│       │   │   ├── layout_agent.py       # 排版输出智能体
│       │   │   ├── polish_article_agent.py# 文章润色智能体
│       │   │   └── review_article_agent.py# 文章审核智能体
│       │   │
│       │   ├── __init__.py
│       │   ├── head_agent.py              # 入口路由智能体
│       │   ├── memory_manage_agent.py    # 上下文记忆管理智能体
│       │   ├── outline_generate_agent.py  # 文章大纲生成智能体
│       │   ├── search_issue_agent.py     # 全网信息检索智能体
│       │   ├── search_simplify_agent.py  # 检索结果提炼智能体
│       │   ├── title_decomposer_agent.py # 主题拆解智能体
│       │   └── write_chapter_agent.py    # 章节撰写智能体
│       │
│       ├── tools/                     # 智能体工具集（可扩展工具）
│       ├── orchestrator.py            # 工作流编排器（LangGraph核心）
│       ├── state_model.py             # 全局状态管理模型
│       │
│       ├── prompts/                   # 智能体Prompt模板库
│       │   ├── __init__.py
│       │   ├── load_prompt.py         # Prompt加载工具
│       │   ├── controller_prompt.yaml
│       │   ├── head_prompt.yaml
│       │   ├── layout_prompt.yaml
│       │   ├── memory_manage_prompt.yaml
│       │   ├── outling_generate_prompt.yaml
│       │   ├── polish_artical_prompt.yaml
│       │   ├── review_artical_prompt.yaml
│       │   ├── search_simplify_prompt.yaml
│       │   ├── title_decomposer.yaml
│       │   └── write_chapter_prompt.yaml
│       │
│       └── utils/                     # 通用工具函数
│           ├── __init__.py
│           ├── file_reader.py         #文件读写
│           └── logs.py                #日志配置
└── test/                              # 单元测试目录
```

## 🚀 快速开始

### 环境准备

确保本地安装 Python 3.12+ 版本，推荐使用虚拟环境隔离依赖。

### 安装依赖

本项目使用 `uv` 作为轻量、跨平台的依赖管理工具：

1. 安装 uv

   ```
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   

2. 安装项目依赖

   ```
   uv sync
   ```

   

### 配置环境变量

复制示例配置文件并填充关键信息（API Key、数据库地址等）：

```
cp .env.example .env  # 占位符：确保项目根目录有.env.example模板文件，补充需要配置的环境变量（如大模型API Key、ES/ChromaDB地址、Travily API Key等）
```

编辑 `.env` 文件，填写以下核心配置（示例）：

```
#modelscope平台
LLM_BASE_URL=https://api.example.com/v1
LLM_BASE_API_KEY=your-api-key-here

#嵌入模型
EMBEDDING_BASE_URL=https://api.example.com/v1
EMBEDDING_BASE_API_KEY=your-embedding-api-key-here

#图片模型
IMAGE_BASE_URL=https://api.example.com/
IMAGE_BASE_API_KEY=your-image-api-key-here

#搜索
TAVILY_API_KEY=your-tavily-api-key-here

# Elasticsearch Settings
ES_USERNAME=elastic
ES_PASSWORD=your-es-password-here

LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your-langsmith-api-key-here" 
LANGSMITH_PROJECT = "OmniWriter"
```

### 运行项目

```
# 基础运行命令
python main.py --topic "文章主题" --demand "用户的额外需求，如（文章风格，字数等）"
```



## 🤝 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

1. 提交issue报告bug或建议新功能
2. 提交pull request改进代码
3. 完善文档

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

## 📞 联系方式

如有任何问题或建议，请通过以下方式反馈：

- **GitHub Issues**：请在项目仓库中提交Issue，这是最推荐的问题反馈方式
- 项目主页：https://github.com/tom946com/OmniWriter

------

<p align="center">Made with ❤️ by the OmniWriter Team</p>
