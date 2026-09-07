from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# 1. 定义输出结构
class SocialMediaPost(BaseModel):
    title: str = Field(description="笔记的标题")
    content: str = Field(description="笔记的正文")
    tags: List[str] = Field(description="3‑5个话题标签")

parser = PydanticOutputParser(pydantic_object=SocialMediaPost)

# 2. 使用ChatPromptTemplate，适配ChatOllama（区分system、user角色）
template = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深的小红书博主，擅长用生动、亲切的语言种草产品。
只输出纯JSON，不要任何解释、思考、markdown代码块。
JSON字段说明：
title：笔记标题（字符串）
content：笔记正文（字符串）
tags：话题标签，3‑5个，字符串数组"""),
    ("human", """请根据以下信息生成一篇小红书笔记文案：
产品名称：{product}
目标受众：{target_audience}
核心卖点：{selling_point}
要求：
1. 标题要吸引眼球，包含1‑2个emoji。
2. 正文需包含个人使用体验，突出核心卖点。
3. 结尾加上3‑5个相关话题标签。
4. 总字数控制在150字左右。""")
])

# ===================== 模型切换示例 =====================
# 示例A：普通无思考模型 qwen2.5:1.5b
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0.3,
    timeout=120
)

# 示例B：Qwen3.5带思考模型，extra_body透传 think=False关闭思考
# llm = ChatOllama(
#     model="qwen3.5:2b",
#     temperature=0.3,
#     timeout=120,
#     extra_body={"think": False}   # ollama顶层参数关闭思考
# )

# 示例C：如果想保留思考过程，可以不设置think=False，从additional_kwargs读取思考内容
# llm = ChatOllama(model="qwen3.5:2b", temperature=0.3, timeout=120)

# 构建链：prompt -> chatollama
chain = template | llm

input_vars = {
    "product": "XX牌无线降噪耳机",
    "target_audience": "通勤上班族",
    "selling_point": "降噪优秀，续航40小时"
}

# invoke 返回 AIMessage 对象
ai_msg = chain.invoke(input_vars)

print("=== 完整AIMessage对象 ===")
# ✅ 修复：reasoning_content 存放在 additional_kwargs，不是顶层属性
reasoning_content = ai_msg.additional_kwargs.get("reasoning_content")
print(f"reasoning_content(思考): {reasoning_content}")

# 模型最终输出内容
raw_response = ai_msg.content
print(f"\n=== 模型原始输出content ===\n{raw_response}\n")

# JSON解析
try:
    result = parser.parse(raw_response)
    print("✅ 解析成功：")
    print(f"标题：{result.title}")
    print(f"正文：{result.content}")
    print(f"标签：{', '.join(result.tags)}")
except Exception as e:
    print(f"❌ 解析失败：{e}")
    print("原始文本：", raw_response)
