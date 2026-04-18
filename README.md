<h1 align="center">OmniWriter: Production-Grade Multi-Agent Article Generation System Based on LangGraph + DeepAgents</h1>

<p align="center">
   Languages: 
   English.
   <a href="./README_CN.md">简体中文.</a>
</p>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
## 📸 Core Features

- 🧠 **Multi-agent Collaboration**: Orchestrate workflows with LangGraph, handle post-processing via DeepAgents, featuring clear division of labor and high scalability
- 🎯 **Hallucination Suppression**: Multi-layer safeguards including structured retrieval term extraction, precise prompt constraints, Rerank retrieval filtering, and passive + active retrieval
- 🔍 **Efficient Retrieval**: Hybrid mode of keyword retrieval (ES) + vector retrieval (ChromaDB), optimized with precise triple retrieval terms and metadata filtering
- 📦 **Context Management**: Phased compression, virtual file system persistence, and automatic message collation by DeepAgents to avoid context overflow
- ⚡ **Fault Tolerance Mechanism**: Retry strategies, automatic retry for format errors, dedicated exception logging nodes, enhancing system stability
- 🚀 **Parallel Processing**: Parallel retrieval for subtasks, parallel chapter writing, and parallel image generation, significantly improving generation efficiency
- 📋 **Full Process Controllability**: Outline interruption review, multi-round iterative optimization, and user-defined modifications to meet personalized needs

## 🛠️ Technology Stack

| Category              | Technology / Framework                                       |
| --------------------- | ------------------------------------------------------------ |
| Programming Language  | Python 3.12+                                                 |
| Agent Framework       | LangGraph (Workflow Orchestration), DeepAgents (Multi-agent Collaboration) |
| Database              | ChromaDB (Vector Database), Elasticsearch (Full-Text Search) |
| Model Monitoring      | LangSmith                                                    |
| Web Retrieval         | Tavily API                                                   |
| Network Requests      | requests, aiohttp                                            |
| Dependency Management | uv                                                           |
| Logging System        | Python logging (Custom Configuration)                        |

## 📊 Core Workflow

OmniWriter breaks down article generation into 5 core stages, with full-process automation and iterability:

### Stage 1: Task Decomposition & Panoramic Retrieval

- Topic Decomposition: Split user-input topics into non-overlapping subtasks
- Parallel Search: Multi-agent pulls raw data from the entire network via coroutine mechanism
- Vector Database Construction: Chunk segmentation + vectorization, stored in ChromaDB (in-memory temporary database) + ES, with subtask metadata tagging

### Stage 2: Material Refinement & Skeleton Planning

- Subtask Result Condensation: Structured extraction (summary + logic + compressed retrieval term list)
- Global Outline Generation: Plan article skeleton based on condensed materials, supporting user review/modification
- Interruption Node: Outline saved, process resumes after user confirmation

### Stage 3: Parallel Drafting & Global Stitching

- Parallel Chapter Writing: Multiple writing agents compose different chapters simultaneously, with targeted material retrieval before writing to prevent hallucinations
- Chapter Stitching: Dedicated node splices complete articles and persists them to the user's virtual file system

### Stage 4: Post-Processing (DeepAgents)

The main agent schedules sub-agents to complete full-dimensional optimization:

- Polishing Agent: Content optimization + supplement of image/code descriptions
- Review Agent: 100-point scoring review; scores below 80 trigger re-polishing
- Image Generation Agent: Parallel image generation, saved to user directory
- Layout Agent: Placeholder replacement + formatting on demand (Markdown by default)

### Stage 5: Iterative Optimization

Users provide revision feedback, and the routing agent judges and executes corresponding nodes directionally (no need to re-run the entire process)

## 🎯 Key Problem Solutions

| Core Problem           | Solution                                                     |
| ---------------------- | ------------------------------------------------------------ |
| LLM Hallucinations     | Compressed retrieval term list + precise prompt constraints + Rerank retrieval filtering + passive/active retrieval supplementation |
| Poor Retrieval Effect  | Precise triple retrieval terms + metadata filtering + ES+ChromaDB hybrid retrieval + raw material fallback |
| Context Overflow       | Phased compression + virtual file system persistence + DeepAgents automatic message collation |
| System Fault Tolerance | Recoverable error retries + LLM retry for format errors + dedicated exception handling nodes + complete log records |

## 📂 Project Structure

```
OmniWriter/
├── .env                               # Environment variable configuration file (API Key, etc.)
├── .gitignore                         # Git ignore file configuration
├── README.md                          # Project documentation
├── uv.lock                            # UV dependency lock file
├── pyproject.toml                     # Python project dependencies & configuration
├── main.py                            # Project entry file
├── data/
│   ├── log/                           # Log files
│   ├── chromadb/                      # Vector database storage
│   └── {user_id}/                     # User file system
│       ├── images/                    # Generated images
│       ├── skills/                    # User-defined skills
│       ├── supplementary_material     # User-provided materials
│       ├── article.txt                # Generated article
│       └── outline.txt                # Generated outline
│       
├── src/
│   ├── core/                          # Core client modules (vector database, search engine, LLM)
│   │   ├── __init__.py
│   │   ├── chroma_client.py           # Chroma vector database client
│   │   ├── es_client.py               # Elasticsearch search engine client
│   │   └── model_client.py            # LLM API client wrapper
│   │
│   └── pipeline/                      # Multi-agent workflow core module
│       ├── agents/                    # All agent implementations
│       │   ├── deepagent/             # DeepAgents submodule
│       │   │   ├── __init__.py
│       │   │   ├── controller_agent.py    # Process control agent
│       │   │   ├── draw_images_agent.py   # Image generation agent
│       │   │   ├── layout_agent.py       # Layout & output agent
│       │   │   ├── polish_article_agent.py# Article polishing agent
│       │   │   └── review_article_agent.py# Article review agent
│       │   │
│       │   ├── __init__.py
│       │   ├── head_agent.py              # Entry routing agent
│       │   ├── memory_manage_agent.py    # Context memory management agent
│       │   ├── outline_generate_agent.py  # Article outline generation agent
│       │   ├── search_issue_agent.py     # Global web information retrieval agent
│       │   ├── search_simplify_agent.py  # Retrieval result refinement agent
│       │   ├── title_decomposer_agent.py # Topic decomposition agent
│       │   └── write_chapter_agent.py    # Chapter writing agent
│       │
│       ├── tools/                     # Agent toolset (extensible tools)
│       ├── orchestrator.py            # Workflow orchestrator (LangGraph core)
│       ├── state_model.py             # Global state management model
│       │
│       ├── prompts/                   # Agent prompt template library
│       │   ├── __init__.py
│       │   ├── load_prompt.py         # Prompt loading utility
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
│       └── utils/                     # Common utility functions
│           ├── __init__.py
│           ├── file_reader.py         # File read/write utilities
│           └── logs.py                # Log configuration
└── test/                              # Unit test directory
```

## 🚀 Quick Start

### Environment Preparation

Ensure Python 3.12+ is installed locally; it is recommended to use a virtual environment to isolate dependencies.

### Install Dependencies

This project uses `uv` as a lightweight, cross-platform dependency management tool:

1. Install uv

   ```
   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies

   ```
   uv sync
   ```

### Configure Environment Variables

Copy the example configuration file and fill in key information (API Key, database address, etc.):

```
cp .env.example .env  # Placeholder: Ensure there is an .env.example template file in the project root directory, and supplement the required environment variables (e.g., LLM API Key, ES/ChromaDB address, Tavily API Key, etc.)
```

Edit the `.env` file and fill in the following core configurations (example):

```
# ModelScope Platform
LLM_BASE_URL=https://api.example.com/v1
LLM_BASE_API_KEY=your-api-key-here

# Embedding Model
EMBEDDING_BASE_URL=https://api.example.com/v1
EMBEDDING_BASE_API_KEY=your-embedding-api-key-here

# Image Model
IMAGE_BASE_URL=https://api.example.com/
IMAGE_BASE_API_KEY=your-image-api-key-here

# Search
TAVILY_API_KEY=your-tavily-api-key-here

# Elasticsearch Settings
ES_USERNAME=elastic
ES_PASSWORD=your-es-password-here

LANGSMITH_TRACING = "true"
LANGSMITH_API_KEY = "your-langsmith-api-key-here" 
LANGSMITH_PROJECT = "OmniWriter"
```

### Run the Project

```
# Basic run command
python main.py --topic "Article Topic" --demand "Additional user requirements (e.g., article style, word count, etc.)"
```

## 🤝 Contribution Guidelines

We welcome contributions in various forms, including but not limited to:

1. Submitting issues to report bugs or suggest new features
2. Submitting pull requests to improve code
3. Improving documentation

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For any questions or suggestions, please provide feedback through the following channels:

- **GitHub Issues**: Submit an issue in the project repository (the most recommended way to report problems)
- Project Homepage: https://github.com/tom946com/OmniWriter

------

<p align="center">Made with ❤️ by the OmniWriter Team</p>

