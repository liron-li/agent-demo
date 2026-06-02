# AI Agent Demo

本项目包含多种 AI Agent 模式的实现示例，用于学习和演示不同的大语言模型智能体架构。

## 项目结构

| 文件 | 模式 | 描述 |
|------|------|------|
| `main.py` | ReAct（基础版） | 梦幻西游热梗二创台词生成 Agent：搜索当前热梗和场景网页灵感、整理素材卡、迁移爆款二创模板、生成并自检可直接配音的台词 |
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
SCRIPT_OUTPUT_DIR="outputs"  # main.py 生成剧本项目的本地保存目录，可选
```

### 2. 安装依赖

```bash
pip install openai python-dotenv google-search-results autogen-agentchat autogen-ext
```

### 3. 运行 Demo

```bash
# 梦幻西游热梗二创台词生成 Agent
python main.py

# 也可以传入更具体的创作需求
python main.py "写一组60秒梦幻西游鉴定无级别翻车的抖音热梗二创台词"
python main.py "写一组梦幻西游新区排队、藏宝阁看号、炼妖打书结合热梗的短视频台词"
python main.py "参考小明剑魔look in my eyes这种贴脸质问模板，写一组梦幻西游藏宝阁看号破防台词"

# 直接传入模板台词，agent 会保留节奏并改写成梦幻西游原创台词
python main.py "写一组梦幻西游藏宝阁看号破防台词" --template "看着我，回答我。你告诉我，这叫毕业号吗？问题到底出在哪里？"

# 多行模板台词建议放到文本文件里
python main.py "写一组梦幻西游炼妖打书破防台词" --template-file ".\templates\look_template.txt"

# ReAct 搜索智能体
python react.py

# Plan and Solve 规划执行
python plan_and_solve.py

# Reflection 反思式代码生成
python reflection.py

# 多智能体协作开发
python autogen.py
```

`main.py` 会按以下链路生成台词：场景识别 → 场景网页灵感搜索 → 当前热梗搜索 → 素材卡整理 → 爆款二创模板迁移 → 玩家梗匹配 → 台词生成 → 质量自检，不合格会自动要求模型重写一次。

输出默认是独角戏台词，不分“主角/队友/系统提示”等角色，适合固定素材换脸、换口播台词的二创形式。

每次生成完成后，会在 `outputs/时间戳_标题或需求/` 下创建独立项目目录，包含 `script.md`、`request.txt`、`review.txt`，如果传入了模板台词还会保存 `template_lines.txt`，并预建 `assets/`、`audio/`、`video/`、`exports/`，方便继续放固定素材、配音和成片。

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
