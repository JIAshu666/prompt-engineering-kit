# evaluator.py
import json
import re
import time
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# ---------- 评审提示词（强化版：明确要求纯 JSON） ----------
JUDGE_TEMPLATE = """
你是一位严格的文案评审专家。请根据以下标准，对AI生成的小红书笔记进行打分（每项1-10分）。

【评分标准】
- format_score: 是否严格遵循JSON格式，字段是否完整
- completeness_score: 是否覆盖产品名称、受众、卖点
- style_score: 是否符合小红书亲切、活泼的风格
- creativity_score: 标题和文案是否吸引人

【用户原始需求】
产品：{product}
受众：{target_audience}
卖点：{selling_point}

【AI生成的回答】
{ai_output}

【重要】请只输出一个合法的JSON对象，不要添加任何解释、前缀或后缀。格式必须如下：
{{
    "format_score": 数字,
    "completeness_score": 数字,
    "style_score": 数字,
    "creativity_score": 数字,
    "overall_score": 数字,
    "feedback": "简要评语"
}}
"""

# ---------- 智能 JSON 提取器 ----------
def extract_json(text: str) -> dict:
    """
    从可能包含额外文字的文本中提取第一个完整的JSON对象。
    支持：纯JSON、markdown代码块、被大括号包围的内容。
    """
    # 去除首尾空白
    text = text.strip()
    
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except:
        pass
    
    # 2. 尝试提取 ```json ... ``` 代码块
    json_block = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except:
            pass
    
    # 3. 尝试提取第一个完整的 {...} 结构（使用正则平衡匹配可能不完美，但适用于简单情况）
    # 使用简单贪心：找第一个 { 和最后一个 } 之间的内容
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            return json.loads(candidate)
        except:
            pass
    
    # 4. 尝试在文本中查找任何类似JSON的片段（不完美，但作为最后手段）
    # 这里可以加更多启发式，但先返回空
    return {}

# ---------- 评审函数（带重试和智能提取） ----------
def evaluate_case(product: str, target_audience: str, selling_point: str,
                  ai_output: str, model: str = "qwen2.5:1.5b", max_retries: int = 2) -> dict:
    """
    调用评审模型打分，从输出中智能提取JSON。
    """
    prompt = PromptTemplate(
        template=JUDGE_TEMPLATE,
        input_variables=["product", "target_audience", "selling_point", "ai_output"]
    )
    filled = prompt.format(
        product=product,
        target_audience=target_audience,
        selling_point=selling_point,
        ai_output=ai_output[:2000]  # 限制长度，防止上下文过长
    )
    
    llm = ChatOllama(model=model, temperature=0.0)
    
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(filled)
            raw = response.content
            
            # 尝试提取JSON
            result = extract_json(raw)
            if result:
                # 确保所有必要字段存在
                required_keys = ["format_score", "completeness_score", "style_score",
                                 "creativity_score", "overall_score", "feedback"]
                for key in required_keys:
                    if key not in result:
                        result[key] = 0 if key != "feedback" else "缺失"
                # 确保所有分数是数字
                for key in ["format_score", "completeness_score", "style_score", "creativity_score", "overall_score"]:
                    if not isinstance(result.get(key), (int, float)):
                        result[key] = 0
                return result
            else:
                # 未提取到有效JSON，抛出异常进入重试
                raise ValueError("无法从评审输出中提取JSON")
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠️ 评审调用失败 ({attempt+1}/{max_retries+1})，重试中...")
                time.sleep(1)
            else:
                print(f"  ❌ 评审最终失败: {e}")
                # 返回默认值
                return {
                    "format_score": 0,
                    "completeness_score": 0,
                    "style_score": 0,
                    "creativity_score": 0,
                    "overall_score": 0,
                    "feedback": "评审失败"
                }