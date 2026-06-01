# AI Agent Demo

本项目包含多种 AI Agent 模式的实现示例，用于学习和演示不同的大语言模型智能体架构。

## 项目结构

| 文件 | 模式 | 描述 |
|------|------|------|
| `main.py` | ReAct（基础版） | 梦幻西游热梗二创短视频脚本生成 Agent：搜索当前热梗和场景网页灵感、匹配玩家梗、生成可拍摄分镜脚本 |
| `react.py` | ReAct | 带 SerpApi 网页搜索的 ReAct 智能体，可回答实时/事实类问题 |
| `plan_and_solve.py` | Plan and Solve | 先规划后执行：将复杂问题分解为步骤列表，再逐步执行 |
| `reflection.py` | Reflection | 反思式代码生成：通过「生成 → 反思 → 优化」迭代提升代码质量 |
| `autogen.py` | Multi-Agent | 多智能体协作：产品经理、工程师、代码审查员协作完成开发任务 |
| `llm_client.py` | - | 通用 LLM 客户端，支持 OpenAI 兼容接口和流式输出 |

## 快速开始

### 1. 环境配置

复制环境变量模板并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入必要的 API 密钥：

```env
# LLM 服务（必需，所有 Demo 都需要）
LLM_API_KEY=""
LLM_MODEL_ID="qwen-max"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_TIMEOUT="60"

# 各 Demo 依赖的 API（按需配置）
SERPAPI_API_KEY=""   # main.py 热梗与场景灵感搜索 / react.py 网页搜索
```

### 2. 安装依赖

```bash
pip install openai python-dotenv google-search-results autogen-agentchat autogen-ext
```

### 3. 运行 Demo

```bash
# 梦幻西游热梗二创短视频脚本生成 Agent
python main.py

# 也可以传入更具体的创作需求
python main.py "写一个60秒梦幻西游鉴定无级别翻车的抖音热梗二创脚本"
python main.py "写一个梦幻西游新区排队、藏宝阁看号、炼妖打书结合热梗的短视频脚本"

# ReAct 搜索智能体
python react.py

# Plan and Solve 规划执行
python plan_and_solve.py

# Reflection 反思式代码生成
python reflection.py

# 多智能体协作开发
python autogen.py
```

## 外部服务依赖

| 服务 | 用途 | 获取方式 |
|------|------|----------|
| **SerpApi** | main.py 当前热梗与场景灵感搜索、react.py 网页搜索 | [serpapi.com](https://serpapi.com) |

## 模式说明

### ReAct

- **Thought**：分析问题、规划下一步
- **Action**：调用工具或输出最终答案
- **Observation**：工具执行结果

### Plan and Solve

- **Planner**：将复杂问题分解为可执行步骤列表
- **Executor**：按顺序执行每个步骤，汇总结果

### Reflection

- **Execution**：根据任务生成初始代码
- **Reflection**：对代码进行评审，找出算法/效率问题
- **Refinement**：根据反馈改进代码，迭代直至满足要求

### Multi-Agent (AutoGen)

- **ProductManager**：需求分析、技术规划
- **Engineer**：编写实现代码
- **CodeReviewer**：代码审查与质量建议
- **UserProxy**：代表用户执行测试

## 许可证

MIT
