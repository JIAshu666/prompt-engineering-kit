# batch_test.py
import json
import time
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from evaluator import evaluate_case   # 假设 evaluator.py 存在且正确

# ============================================================
# 1. 定义输出数据结构（与 main.py 保持一致）
# ============================================================
class SocialMediaPost(BaseModel):
    title: str = Field(description="笔记的标题，必须吸引人")
    content: str = Field(description="笔记的正文内容，包含个人体验")
    tags: List[str] = Field(description="3-5个相关话题标签的列表")

# ============================================================
# 2. 创建解析器
# ============================================================
parser = PydanticOutputParser(pydantic_object=SocialMediaPost)

# ============================================================
# 3. 提示词模板（与 main.py 保持一致）
# ============================================================
TEMPLATE_STR = """
你是一位资深的小红书博主，擅长用生动、亲切的语言种草产品。

请根据以下信息，为产品生成一篇小红书笔记文案：

产品名称：{product}
目标受众：{target_audience}
核心卖点：{selling_point}

要求：
1. 标题要吸引眼球，包含1-2个emoji。
2. 正文需包含个人使用体验，突出核心卖点。
3. 结尾加上3-5个相关话题标签。
4. 总字数控制在150字左右。

{format_instructions}

请直接输出 JSON 格式的结果，不要有任何多余的解释或 Markdown 标记。
"""

prompt = PromptTemplate(
    template=TEMPLATE_STR,
    input_variables=["product", "target_audience", "selling_point"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ============================================================
# 4. 生成函数（使用 ChatOllama）
# ============================================================
def generate_post(case: dict, model: str = "qwen2.5:1.5b") -> str:
    """生成文案，返回原始输出字符串"""
    filled_prompt = prompt.format(**case)
    llm = ChatOllama(model=model, temperature=0.3)
    response = llm.invoke(filled_prompt)
    return response.content

# ============================================================
# 5. 带重试的解析函数
# ============================================================
def parse_with_retry(raw_output: str, max_retries: int = 2) -> Optional[SocialMediaPost]:
    cleaned = raw_output.strip()
    # 尝试提取 JSON 代码块
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    
    for attempt in range(max_retries + 1):
        try:
            return parser.parse(cleaned)
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⚠️ 解析尝试 {attempt+1} 失败，重试中...")
                # 尝试移除前后缀
                cleaned = re.sub(r'^[^{]*', '', cleaned)
                cleaned = re.sub(r'[^}]*$', '', cleaned)
                time.sleep(1)
            else:
                print(f"  ❌ 解析失败，已重试 {max_retries} 次")
                return None
    return None

# ============================================================
# 6. 批量测试主函数
# ============================================================
def run_batch_test(test_file: str = "test_cases.json", 
                   generation_model: str = "qwen2.5:1.5b") -> List[Dict[str, Any]]:
    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    results = []
    total = len(cases)
    
    for idx, case in enumerate(cases, 1):
        print(f"\n🔄 测试用例 {idx}/{total}: {case['product']}")
        
        # 1. 生成
        start_time = time.time()
        raw_output = generate_post(case, model=generation_model)
        elapsed = time.time() - start_time
        
        # 2. 解析（带重试）
        parsed = parse_with_retry(raw_output)
        parse_success = parsed is not None
        
        # 3. 评审打分
        score_result = evaluate_case(
            case["product"], 
            case["target_audience"], 
            case["selling_point"], 
            raw_output
        )
        
        # 4. 记录
        results.append({
            "case": case,
            "raw_output": raw_output,
            "parsed": parsed.dict() if parsed else None,
            "parse_success": parse_success,
            "elapsed_seconds": round(elapsed, 2),
            "judge_scores": score_result
        })
        
        status = "✅" if parse_success else "❌"
        print(f"  {status} 解析: {'成功' if parse_success else '失败'}, 综合评分: {score_result.get('overall_score', 'N/A')}")
        time.sleep(1)
    
    return results

# ============================================================
# 7. 生成报告
# ============================================================
def generate_report(results: List[Dict[str, Any]]):
    total = len(results)
    if total == 0:
        print("没有测试结果")
        return
    
    success_count = sum(1 for r in results if r["parse_success"])
    avg_overall = sum(r["judge_scores"].get("overall_score", 0) for r in results) / total
    
    print("\n" + "="*50)
    print(f"📊 批量测试报告")
    print("="*50)
    print(f"总用例数：{total}")
    print(f"解析成功率：{success_count}/{total} ({success_count/total*100:.1f}%)")
    print(f"平均综合得分：{avg_overall:.1f}/10")
    print("\n各用例详情：")
    for i, r in enumerate(results, 1):
        status = "✅" if r["parse_success"] else "❌"
        score = r["judge_scores"].get("overall_score", "N/A")
        print(f"  {status} 用例{i}: {r['case']['product']} | 得分: {score} | 耗时: {r['elapsed_seconds']}s")
    print("="*50)

# ============================================================
# 8. 主入口
# ============================================================
if __name__ == "__main__":
    results = run_batch_test(generation_model="qwen2.5:1.5b")
    generate_report(results)
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存至 test_results.json")