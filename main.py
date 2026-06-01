import io
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
from serpapi import SerpApiClient

# 加载 .env 文件中的环境变量
load_dotenv()


# --- 1. 配置 LLM 客户端 ---
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL_ID")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# 强制将标准输出流的编码设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


AGENT_SYSTEM_PROMPT = """
你是一个“梦幻西游热梗二创短视频编剧 Agent”。你的任务不是简单写段子，
而是像短视频编剧总监一样：先分析主题和玩家痛点，匹配梦幻西游具体场景，
再调用外部搜索获取该场景的玩家讨论/攻略痛点/翻车案例，同时了解当前中文互联网/短视频热梗，
最后筛选适合改编的梗，把它们转译成梦幻西游玩家能秒懂、能拍出来、能引发评论互动的原创搞笑脚本。

# 创作定位
- 面向熟悉《梦幻西游》的玩家，风格轻松、沙雕、反转、带玩家共鸣和吐槽欲。
- 优先使用具体玩法语境，例如鉴定装备、炼妖打书、抓鬼、副本、帮战、比武、摆摊、跑商、五开、师门、跑环、门派转换、藏宝阁、帮派社交等。
- 不冒充官方，不复刻官方剧情或大段原文台词；输出应是原创同人向搞笑短视频脚本。
- 热梗必须二创成梦幻西游语境，不能生搬硬套，不能只把流行语贴到台词里。
- 梗要服务剧情，避免只堆网络热词；每个热梗都要对应明确的游戏场景或玩家痛点。
- 台词要短、狠、好配音；画面要能用游戏录屏、聊天框、角色动作、弹幕、音效完成。

# 热梗二创方法
1. 先用 `analyze_script_request` 匹配具体梦幻西游场景。
2. 再用 `search_scene_inspiration` 搜该场景的玩家讨论、攻略痛点、翻车案例，提炼真实玩家情绪。
3. 再从热梗搜索结果中挑选 2-3 个“表达结构”或“情绪模板”，不要照抄原梗。
4. 把热梗翻译成梦幻西游场景：谁遇到了什么玩法痛点，为什么误会升级，谁在最后补刀。
5. 每个梗都要落到一种玩家情绪：嘴硬、破防、赌狗心理、攀比、社恐组队、老板沉默、队长崩溃、五开手忙脚乱。
6. 用“三段式”制造笑点：预期很满 -> 过程翻车 -> 结果反向封神。
7. 最终脚本宁可少用热梗，也要保证剧情清楚、反转明确、玩家能看懂。

# 可用工具
- `analyze_script_request(request: str)`: 分析用户需求，提取主题、场景、时长、平台、笑点方向。
- `search_scene_inspiration(scene: str, topic: str)`: 调用外部搜索，获取该梦幻西游场景的网页资料、玩家讨论和翻车灵感。
- `search_current_hot_memes(topic: str, platform: str)`: 调用外部搜索，获取当前短视频/中文互联网热梗和流行表达。
- `get_mhxy_comedy_pack(scene: str, style: str)`: 获取适合指定场景和风格的玩家梗、冲突和包袱素材。
- `get_short_video_structure(duration: str, platform: str)`: 获取短视频节奏结构和分镜格式建议。

# 最终答案必须包含
1. 标题
2. 一句话卖点：用一句短视频标题党式文案说明为什么值得看
3. 核心梗概
4. 场景灵感来源：概括网页搜索到的玩家痛点或玩法矛盾，不要复制网页原文
5. 热梗二创说明：列出 2-3 个热梗/情绪模板如何被改造成梦幻西游梗
6. 人物设定
7. 分镜脚本：镜头 / 画面动作 / 台词或旁白 / 字幕音效 / 时长
8. 评论区钩子：给一个能引导玩家留言的开放问题
9. 拍摄或剪辑提示

# 输出格式要求
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示
- 每次只输出一对 Thought-Action。
- Action 必须在同一行，不要换行。
- 匹配到场景后，必须调用一次 `search_scene_inspiration` 获取场景网页灵感。
- 生成最终脚本前，必须至少调用一次 `search_current_hot_memes`。
- 网页搜索结果只作为灵感，不要复述长段网页内容，不要编造“官方结论”。
- 如果搜索结果质量一般，要提炼“情绪模板”而不是强行引用具体热梗。
- 工具信息足够后，必须用 Action: Finish[最终答案] 输出完整脚本。
- 最终脚本用中文输出，适合 30-90 秒短视频。
- 最终脚本不要解释你作为 Agent 的思考过程，只给可执行成片内容。

请开始吧！
"""


@dataclass
class ScriptBrief:
    theme: str
    scene: str
    duration: str
    platform: str
    style: str
    audience: str


def analyze_script_request(request: str) -> str:
    """
    分析短视频脚本需求，给模型一个稳定的创作简报。
    """
    text = request.strip()

    scene_keywords = {
        "鉴定": "鉴定装备",
        "无级别": "鉴定装备",
        "军火": "鉴定装备",
        "装备": "鉴定装备",
        "抓鬼": "抓鬼组队",
        "组队": "组队翻车",
        "五开": "五开日常",
        "师门": "师门任务",
        "跑环": "跑环破防",
        "传说": "跑环破防",
        "摆摊": "摆摊砍价",
        "摊位": "摆摊砍价",
        "砍价": "摆摊砍价",
        "宝宝": "炼妖打书",
        "打书": "炼妖打书",
        "炼妖": "炼妖打书",
        "合宠": "炼妖打书",
        "帮战": "帮战现场",
        "比武": "比武大会",
        "PK": "比武大会",
        "pk": "比武大会",
        "华山": "比武大会",
        "副本": "副本翻车",
        "乌鸡": "副本翻车",
        "车迟": "副本翻车",
        "神器": "神器任务",
        "地煞": "地煞挑战",
        "天罡": "地煞挑战",
        "封妖": "封妖翻车",
        "捉妖": "封妖翻车",
        "跑商": "跑商压价",
        "商人": "跑商压价",
        "藏宝阁": "藏宝阁看号",
        "买号": "藏宝阁看号",
        "卖号": "藏宝阁看号",
        "转门派": "门派转换",
        "门派转换": "门派转换",
        "新区排队": "新区排队",
        "服务器排队": "新区排队",
        "转区": "转区排队",
        "排队": "转区排队",
        "帮派": "帮派社交",
        "结拜": "帮派社交",
        "情缘": "情缘翻车",
        "情侣": "情缘翻车",
        "结婚": "情缘翻车",
        "锦衣": "锦衣攀比",
        "祥瑞": "锦衣攀比",
        "限量": "锦衣攀比",
        "打图": "打图玄学",
        "宝图": "打图玄学",
        "挖宝": "挖宝翻车",
        "高级藏宝图": "挖宝翻车",
        "科举": "科举答题",
        "庭院": "庭院种树",
        "牧场": "牧场经营",
        "口袋版": "口袋版上头",
        "新区": "新区排队",
        "服务器": "新区排队",
    }
    scene = next(
        (
            value
            for key, value in sorted(scene_keywords.items(), key=lambda item: len(item[0]), reverse=True)
            if key in text
        ),
        "玩家日常破防",
    )

    duration_match = re.search(r"(\d{2,3})\s*秒", text)
    duration = f"{duration_match.group(1)}秒" if duration_match else "60秒"

    platform = "抖音/快手/B站竖屏"
    if "B站" in text or "b站" in text:
        platform = "B站"
    elif "抖音" in text:
        platform = "抖音"
    elif "快手" in text:
        platform = "快手"
    elif "小红书" in text:
        platform = "小红书"

    style = "沙雕反转"
    if "吐槽" in text:
        style = "犀利吐槽"
    elif "无厘头" in text:
        style = "无厘头"
    elif "热血" in text:
        style = "热血反差喜剧"
    elif "情侣" in text or "CP" in text or "cp" in text:
        style = "情侣互坑喜剧"

    brief = ScriptBrief(
        theme=text or "梦幻西游搞笑短视频脚本",
        scene=scene,
        duration=duration,
        platform=platform,
        style=style,
        audience="熟悉梦幻西游玩法、爱看游戏吐槽和反转段子的玩家",
    )

    return (
        "脚本创作简报:\n"
        f"- 用户主题: {brief.theme}\n"
        f"- 推荐场景: {brief.scene}\n"
        f"- 建议时长: {brief.duration}\n"
        f"- 发布平台: {brief.platform}\n"
        f"- 喜剧风格: {brief.style}\n"
        f"- 目标观众: {brief.audience}\n"
        "- 创作原则: 开头3秒必须有钩子，中段制造误会或升级冲突，结尾给反转和评论区互动。\n"
        "- 可选扩展场景: 鉴定装备、炼妖打书、抓鬼组队、副本翻车、地煞挑战、神器任务、"
        "帮战现场、比武大会、跑商压价、藏宝阁看号、门派转换、情缘翻车、锦衣攀比、"
        "新区排队、挖宝翻车、科举答题、庭院牧场等。"
    )


def search_current_hot_memes(topic: str, platform: str) -> str:
    """
    使用 SerpApi 搜索当前热梗，为脚本二创提供实时素材。
    """
    if not SERPAPI_API_KEY:
        return (
            "错误: 未配置 SERPAPI_API_KEY，无法调用外部搜索获取当前热梗。\n"
            "请在 .env 中配置 SERPAPI_API_KEY 后重试。"
        )

    today = datetime.now().strftime("%Y年%m月")
    query = (
        f"{today} 中文互联网 短视频 热梗 流行语 抖音 B站 小红书 "
        f"{platform} {topic}"
    )
    print(f"正在执行 [SerpApi] 热梗搜索: {query}")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "gl": "cn",
        "hl": "zh-cn",
        "num": 8,
    }

    try:
        client = SerpApiClient(params)
        results = client.get_dict()
    except Exception as e:
        return f"错误: 搜索当前热梗时发生异常 - {e}"

    if "error" in results:
        return f"错误: SerpApi 返回异常 - {results['error']}"

    snippets = []
    for index, result in enumerate(results.get("organic_results", [])[:6], start=1):
        title = result.get("title", "").strip()
        snippet = result.get("snippet", "").strip()
        source = result.get("source") or result.get("displayed_link", "")
        if title or snippet:
            snippets.append(f"[{index}] {title}\n来源: {source}\n摘要: {snippet}")

    if not snippets:
        return (
            f"没有搜索到足够明确的当前热梗。查询词: {query}\n"
            "可以使用通用短视频结构，但最终答案要提醒用户热梗素材不足。"
        )

    return (
        f"当前热梗搜索结果（查询时间: {today}，平台: {platform}）:\n"
        + "\n\n".join(snippets)
        + "\n\n二创要求: 从以上结果中提炼 2-3 个适合短视频的热梗表达，"
        "不要照搬原句；把它们改造成梦幻西游的装备、组队、任务、交易、宝宝等玩家场景。"
    )


def search_scene_inspiration(scene: str, topic: str) -> str:
    """
    使用 SerpApi 搜索指定梦幻西游场景的玩家讨论和玩法痛点。
    """
    if not SERPAPI_API_KEY:
        return (
            "错误: 未配置 SERPAPI_API_KEY，无法调用外部搜索获取场景灵感。\n"
            "请在 .env 中配置 SERPAPI_API_KEY 后重试。"
        )

    query = (
        f"梦幻西游 {scene} 玩家吐槽 攻略 痛点 翻车 案例 段子 "
        f"贴吧 论坛 B站 {topic}"
    )
    print(f"正在执行 [SerpApi] 场景灵感搜索: {query}")

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "gl": "cn",
        "hl": "zh-cn",
        "num": 8,
    }

    try:
        client = SerpApiClient(params)
        results = client.get_dict()
    except Exception as e:
        return f"错误: 搜索场景灵感时发生异常 - {e}"

    if "error" in results:
        return f"错误: SerpApi 返回异常 - {results['error']}"

    snippets = []
    for index, result in enumerate(results.get("organic_results", [])[:6], start=1):
        title = result.get("title", "").strip()
        snippet = result.get("snippet", "").strip()
        source = result.get("source") or result.get("displayed_link", "")
        if title or snippet:
            snippets.append(f"[{index}] {title}\n来源: {source}\n摘要: {snippet}")

    if not snippets:
        return (
            f"没有搜索到足够明确的场景灵感。查询词: {query}\n"
            "可以使用内置场景素材包，但最终脚本要减少对具体网页资料的依赖。"
        )

    return (
        f"梦幻西游场景网页灵感（场景: {scene}）:\n"
        + "\n\n".join(snippets)
        + "\n\n提炼要求: 从以上结果中提炼玩家真实痛点、常见误会、翻车瞬间、可视化动作。"
        "只做灵感参考，不要复制网页原文；把资料改写成原创短视频桥段。"
    )


def get_mhxy_comedy_pack(scene: str, style: str) -> str:
    """
    返回梦幻西游玩家能快速共鸣的原创搞笑素材。
    """
    packs: Dict[str, Dict[str, str]] = {
        "鉴定装备": {
            "conflict": "主角扬言今天要逆天改命，结果全队围观他从自信到沉默。",
            "jokes": "逆袭宣言、玄学垫刀、帮派频道立 flag、鉴定结果比系统消息还冷静。",
            "punchline": "最后发现真正的无级别不是装备，是主角的嘴硬没有等级限制。",
        },
        "抓鬼组队": {
            "conflict": "队长说五分钟高效抓鬼，队友却把每一步都玩成突发事件。",
            "jokes": "队长催坐标、队友迷路、宠物比人先躺、挂机的人突然抢戏。",
            "punchline": "最后鬼没抓几个，队友把队长的耐心抓没了。",
        },
        "组队翻车": {
            "conflict": "所有人进队前都说自己很稳，开打后每个人都在证明自己只是说话稳。",
            "jokes": "装备展示只截上半段、语音里全是借口、输出像在给怪按摩。",
            "punchline": "最靠谱的是怪，因为它每回合都准时出手。",
        },
        "五开日常": {
            "conflict": "主角以为五开是掌控全局，实际像同时带五个亲戚出门。",
            "jokes": "窗口切错、号走丢、一个号掉线全家停工、自己跟自己吵架。",
            "punchline": "最后系统没封号，主角先把自己的大脑封印了。",
        },
        "师门任务": {
            "conflict": "主角接师门前信心满满，接完发现师父像在考验亲情。",
            "jokes": "要的物品永远刚卖掉、跑腿路线像环游三界、奖励像安慰奖。",
            "punchline": "师父说锻炼心性，主角说确实，心已经没了。",
        },
        "跑环破防": {
            "conflict": "主角计划低成本跑环，任务链用事实告诉他什么叫预算只是幻觉。",
            "jokes": "传说物品精准狙击钱包、世界频道求助、朋友从鼓励变成围观。",
            "punchline": "最后跑完一环，主角觉得自己才是被跑的那个。",
        },
        "摆摊砍价": {
            "conflict": "买家砍价像做法，卖家守价像守城，双方都觉得自己亏了。",
            "jokes": "老板在吗、诚心要、再便宜点、朋友说这个价不如送我。",
            "punchline": "成交后两人同时发朋友圈: 今天遇到一个不会砍价的。",
        },
        "炼妖打书": {
            "conflict": "主角把宝宝当未来战神培养，结果每一步都像在拆盲盒。",
            "jokes": "技能顶掉核心、玄学改名、围观群众马后炮、宝宝眼神逐渐清澈。",
            "punchline": "最后宝宝没成神，主角成了全区反面教材。",
        },
        "帮战现场": {
            "conflict": "帮主赛前激情动员，开打后频道只剩坐标、救我、谁点错了。",
            "jokes": "战术很丰满、执行很即兴、指挥破音、队友沉默但角色很忙。",
            "punchline": "复盘时大家一致认为输在对面也参加了帮战。",
        },
        "副本翻车": {
            "conflict": "队伍觉得副本闭眼过，结果每个小机制都像突然加试。",
            "jokes": "攻略只看标题、队友各打各的、奶妈比输出还想输出。",
            "punchline": "最后副本没教育角色，教育了屏幕前的人。",
        },
        "神器任务": {
            "conflict": "主角说神器任务稳过，开打后发现真正的神器是队友的借口。",
            "jokes": "攻略看到一半、集合拖延、关键回合点错、语音里全是我以为。",
            "punchline": "最后神器没拿到，队伍拿到了年度甩锅奖。",
        },
        "地煞挑战": {
            "conflict": "队伍进场前像职业战队，第一回合后像临时拼车。",
            "jokes": "封系空封、输出刮痧、辅助疯狂找药、老板开始沉默。",
            "punchline": "地煞没说话，但每一次出手都像在打差评。",
        },
        "封妖翻车": {
            "conflict": "主角说路过顺手封个妖，结果被妖顺手教育做人。",
            "jokes": "顺手变专场、队友临时掉线、宝宝被点名、聊天框突然安静。",
            "punchline": "最后妖没被封住，主角的自信被永久封印。",
        },
        "跑商压价": {
            "conflict": "主角以为自己是商业奇才，路线一跑才发现物价比队友还叛逆。",
            "jokes": "高买低卖、帮派催进度、价格刚刷新就错过、算盘打到自己脸上。",
            "punchline": "最后商没跑明白，帮派资金看了都想报警。",
        },
        "藏宝阁看号": {
            "conflict": "主角预算三千想买毕业号，点开藏宝阁后开始重新定义毕业。",
            "jokes": "只看收藏不下单、截图给朋友鉴宝、卖点全靠情怀、缺点全靠缘分。",
            "punchline": "最后号没买，主角买到了清醒。",
        },
        "门派转换": {
            "conflict": "主角以为转门派能逆天改命，转完发现命没改，钱先改姓了。",
            "jokes": "门派攻略看花眼、装备不兼容、朋友激情推荐、实战原地迷路。",
            "punchline": "最后输出没变高，角色倒是多了一份职业规划焦虑。",
        },
        "转区排队": {
            "conflict": "主角想去新区重新做人，排队界面让他先学会重新做人。",
            "jokes": "倒计时反复横跳、朋友先进服炫耀、验证码像副本机制、心态逐渐离线。",
            "punchline": "最后人没转过去，灵魂已经提前跨服。",
        },
        "帮派社交": {
            "conflict": "新人进帮只想安静养老，帮派频道三分钟把他安排成全能社畜。",
            "jokes": "欢迎新人、立刻求助、帮主画饼、老成员围观认亲。",
            "punchline": "最后主角没找到组织，组织找到了主角的空闲时间。",
        },
        "情缘翻车": {
            "conflict": "主角以为游戏情缘是甜甜恋爱，结果聊天记录更像帮派任务派单。",
            "jokes": "上线先问副本、礼物预算拉满、截图误会、朋友群集体吃瓜。",
            "punchline": "最后情缘没奔现，账单先奔溃。",
        },
        "锦衣攀比": {
            "conflict": "主角说自己不在乎外观，看到限量锦衣后嘴比钱包先投降。",
            "jokes": "只看不买、试穿十分钟、朋友一句挺一般、下一秒付款成功。",
            "punchline": "最后属性没涨，虚荣心得到了满级强化。",
        },
        "打图玄学": {
            "conflict": "主角坚信今天打图爆率拉满，结果系统用沉默回应他的信仰。",
            "jokes": "换地图玄学、换时间玄学、换称谓玄学、最后怀疑自己号不干净。",
            "punchline": "图没打出几张，玄学论文写了三千字。",
        },
        "挖宝翻车": {
            "conflict": "主角拿着高级藏宝图幻想暴富，落铲后发现土地也会讲冷笑话。",
            "jokes": "朋友围观开香槟、坐标神秘、挖前祈祷、挖后沉默。",
            "punchline": "最后宝没挖到，挖出了自己对概率的误解。",
        },
        "科举答题": {
            "conflict": "主角觉得科举是送分活动，题目出来后发现自己只适合送人头。",
            "jokes": "百度还没打开、队友乱报答案、常识突然断线、蒙题全避开正确选项。",
            "punchline": "最后知识没改变命运，倒是改变了主角的自尊。",
        },
        "庭院种树": {
            "conflict": "主角把庭院当休闲养老，结果种树比上班还需要打卡。",
            "jokes": "忘浇水、朋友来蹭、收成像盲盒、庭院管理变绩效考核。",
            "punchline": "最后树成熟了，主角也被生活催熟了。",
        },
        "牧场经营": {
            "conflict": "主角想靠牧场岁月静好，实际每天像在管理一个小型公司。",
            "jokes": "喂养排班、繁殖玄学、好友互访、收益算到头秃。",
            "punchline": "最后动物都很稳定，只有主角情绪不稳定。",
        },
        "口袋版上头": {
            "conflict": "主角说口袋版只是顺手点一下，结果顺手顺到凌晨两点。",
            "jokes": "碎片时间变整块时间、奖励诱惑、再点最后一次、手机电量先投降。",
            "punchline": "最后号没变强多少，拇指练成了门派首席。",
        },
        "新区排队": {
            "conflict": "主角进新区准备冲级，结果第一关是登录排队修心。",
            "jokes": "预约豪言、排队截图、朋友已拜师、自己还在选择服务器。",
            "punchline": "最后别人冲新区，主角冲的是耐心上限。",
        },
    }
    material = packs.get(scene, packs["组队翻车"])
    return (
        f"搞笑素材包（场景: {scene}，风格: {style}）:\n"
        f"- 核心冲突: {material['conflict']}\n"
        f"- 可用笑点: {material['jokes']}\n"
        f"- 结尾包袱: {material['punchline']}\n"
        "- 常用角色: 嘴硬队长、沉默老板、玄学大师、挂机队友、围观好友、突然补刀的系统提示。\n"
        "- 热梗融合建议: 把热梗当作人物口癖、弹幕误会、系统提示或结尾反转，不要让角色只是在念梗。\n"
        "- 表演建议: 台词短、节奏快，每 5-8 秒给一个小笑点，结尾用反差完成记忆点。"
    )


def get_short_video_structure(duration: str, platform: str) -> str:
    """
    返回短视频节奏和分镜结构建议。
    """
    return (
        f"短视频结构建议（{duration}，{platform}）:\n"
        "- 0-3秒: 强钩子，直接抛出离谱承诺或反差结果。\n"
        "- 4-15秒: 建立人物关系和玩法场景，让玩家迅速知道在玩什么梗。\n"
        "- 16-40秒: 冲突连续升级，每个镜头只服务一个笑点。\n"
        "- 41-55秒: 误会揭晓或翻车达到最高点。\n"
        "- 56-60秒: 反转收尾，留下评论区互动问题。\n"
        "- 分镜字段: 镜头编号、画面动作、台词/旁白、字幕音效、预计时长。"
    )


# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "analyze_script_request": analyze_script_request,
    "search_scene_inspiration": search_scene_inspiration,
    "search_current_hot_memes": search_current_hot_memes,
    "get_mhxy_comedy_pack": get_mhxy_comedy_pack,
    "get_short_video_structure": get_short_video_structure,
}


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容 OpenAI 接口的 LLM 服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM API 来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


def extract_action(llm_output: str) -> str | None:
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        return None
    return action_match.group(1).strip()


def run_agent(user_prompt: str, max_steps: int = 7) -> None:
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    prompt_history = [f"用户请求: {user_prompt}"]
    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(max_steps):
        print(f"--- 循环 {i + 1} ---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        # 模型可能会输出多余的 Thought-Action，需要截断
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        action_str = extract_action(llm_output)
        if not action_str:
            observation = "错误: 未能解析到 Action 字段。请确保回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            continue

        if action_str.startswith("Finish"):
            final_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
            final_answer = final_match.group(1).strip() if final_match else action_str
            print(f"任务完成，最终答案:\n{final_answer}")
            break

        tool_match = re.search(r"(\w+)\((.*)\)", action_str, re.DOTALL)
        if not tool_match:
            observation = f"错误: 无法解析工具调用 '{action_str}'"
        else:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

            if tool_name in available_tools:
                try:
                    observation = available_tools[tool_name](**kwargs)
                except TypeError as e:
                    observation = f"错误: 工具参数不匹配 - {e}"
            else:
                observation = f"错误: 未定义的工具 '{tool_name}'"

        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)
    else:
        print("已达到最大循环次数，请增加 max_steps 或让需求更具体。")


def get_user_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return "帮我写一个结合当前热梗的梦幻西游搞笑短视频脚本"


if __name__ == "__main__":
    run_agent(get_user_prompt())
