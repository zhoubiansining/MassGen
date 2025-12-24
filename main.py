import asyncio
import os
from dotenv import load_dotenv
from agents.retrieval_agent import RetrievalAgent

# 加载 .env 文件中的环境变量
load_dotenv()

async def main():
    """
    启动 Retrieval Agent 进行 LLM KV Cache 文献检索。
    """
    
    # ==========================================
    # 1. 配置区域 (Configuration)
    # ==========================================
    
    # 优先检查 DEEPSEEK_API_KEY
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model_name = "deepseek-chat"
    base_url = "https://api.deepseek.com"

    # 如果没有 DeepSeek Key，尝试检查 OPENAI_API_KEY (作为备选)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("⚠️ 未找到 DEEPSEEK_API_KEY,切换为 OpenAI 配置...")
            model_name = "gpt-4o"
            base_url = None # 使用官方默认地址
        else:
            print("❌ 错误: 未在 .env 文件或环境变量中找到 API Key (DEEPSEEK_API_KEY 或 OPENAI_API_KEY)。")
            return

    # ==========================================
    # 2. 初始化 Agent
    # ==========================================
    
    print(f"🤖 正再初始化 RetrievalAgent...")
    print(f"   - Model: {model_name}")
    print(f"   - Base URL: {base_url if base_url else 'Default'}")

    try:
        # 显式传入所有配置，不依赖默认值
        agent = RetrievalAgent(
            model=model_name,
            base_url=base_url,
            api_key=api_key
        )
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        return

    # ==========================================
    # 3. 定义检索任务 (Task Definition)
    # ==========================================
    
    # 针对你要求的 KV Cache 主题，我设计了一个详细的 Research Query
    topic = (
        "Investigate the latest advancements (2023-2025) in LLM KV Cache optimization. "
        "Focus on key techniques such as: "
        "1. KV Cache Compression (Quantization, Sparse Attention). "
        "2. Eviction Policies (e.g., H2O, StreamingLLM, SnapKV). "
        "3. Efficient Memory Management for Long-Context Inference. "
        "Please provide a comprehensive summary of the state-of-the-art methods."
    )
    
    print(f"\n🔍 开始执行检索任务:\n{topic}\n")
    print("-" * 60)

    # ==========================================
    # 4. 运行 Agent (Execution)
    # ==========================================
    
    try:
        # 异步运行 Agent
        final_report = await agent.run(topic)
        
        # ==========================================
        # 5. 输出结果
        # ==========================================
        print("\n" + "="*60)
        print("📝 FINAL LITERATURE REVIEW REPORT")
        print("="*60)
        print(final_report)
        
        # 可选：保存到文件
        with open("review_result.md", "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\n✅ 结果已保存至 review_result.md")
        
    except Exception as e:
        print(f"\n❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())