import sys
import os
import re

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import asyncio
from utils.taobao_crawler import TaobaoCrawler
from utils.data_processor import DataProcessor
from agents.fashion_agent import FashionAgent
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv(override=True)



# 全局变量
crawler = None
data_processor = None
fashion_agent = None
clothing_data = None

MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "AI21-Jamba-1.5-Large",
    "AI21-Jamba-1.5-Mini",
    "Codestral-2501",
    "Cohere-command-r",
    "Ministral-3B",
    "Mistral-Large-2411",
    "Mistral-Nemo",
    "Mistral-small"
]

async def start_crawler():
    """启动爬虫并返回登录页面URL"""
    global crawler
    try:
        crawler = TaobaoCrawler()
        # 直接登录
        if crawler.login():
            return "登录成功！"
        return "登录失败，请重试。"
    except Exception as e:
        print(f"Error in start_crawler: {str(e)}")
        return f"发生错误: {str(e)}"

async def check_login():
    """检查登录状态"""
    if crawler:
        return "已登录"
    return "未登录"

async def process_data():
    """处理爬取的数据"""
    global data_processor, clothing_data
    if not crawler:
        return []
        
    # 初始化数据处理器（放在外面，因为即使没有数据也需要这个对象）
    data_processor = DataProcessor(data_dir="data")
    
    # 获取数据
    items = crawler.get_purchase_history(days=30)
    if items:
        try:
            # 保存原始数据
            crawler.save_to_csv(items, "data/new_taobao_purchases.csv")
            # 将items转换为DataFrame
            items_df = pd.DataFrame(items)
            # 处理数据
            clothing_data = data_processor.process_data(items_df)
            # 保存处理后的数据
            clothing_data.to_csv("data/new_processed_taobao_purchases.csv", index=False, encoding='utf-8-sig')
            # 关闭浏览器
            crawler.close()
            # 返回图片URL列表
            print(get_image_urls())
            return get_image_urls()
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return []
    return []

def get_image_urls():
    """获取所有图片URL"""
    if clothing_data is not None:
        return list(clothing_data['image_url'].values)
    return []

def update_model(selected_model):
    os.environ["GITHUB_MODEL"] = selected_model
    return f"当前选择的模型是：{selected_model}"

async def get_recommendation(style_preference: str, temperature: float = None, mood: str = None):
    """获取服装推荐"""
    global fashion_agent
    if clothing_data is not None:
        try:
            # 初始化推荐代理
            fashion_agent = FashionAgent(clothing_data)
            # 获取推荐
            result = await fashion_agent.process_request(
                style_preference=style_preference,
                temperature=temperature,
                mood=mood
            )
            print("LLM result:",result)
            # 解析推荐结果
            recommended_images = re.findall(r'https?://[^\s]+\.jpg', result)
            print("Extracted image URLs:", recommended_images)
            return recommended_images, result
            # try:
            #     recommendation = json.loads(result)
            #     # 获取推荐服装的图片URL
            #     recommended_images = []
            #     if 'top' in recommendation and 'image_url' in recommendation['top']:
            #         recommended_images.append(recommendation['top']['image_url'])
            #     if 'bottom' in recommendation and 'image_url' in recommendation['bottom']:
            #         recommended_images.append(recommendation['bottom']['image_url'])
            #     return recommended_images, result
            # except json.JSONDecodeError as e:
            #     print(f"Error parsing recommendation: {str(e)}")
            #     return [], "解析推荐结果时出错"
            
            
        except Exception as e:
            print(f"Error in get_recommendation: {str(e)}")
            return [], f"获取推荐时出错: {str(e)}"
    return [], "请先处理数据"

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 服装搭配推荐系统")
    
    with gr.Row():
        with gr.Column():
            # 爬虫部分
            gr.Markdown("## 1. 获取淘宝数据")
            start_button = gr.Button("开始登录")
            login_status = gr.Textbox(label="登录状态", interactive=False)
            process_button = gr.Button("获取服饰数据")
            
            # 显示所有服装图片
            gr.Markdown("## 2. 我的服装")
            gallery = gr.Gallery(label="我的服装", show_label=False)
            
            # 推荐部分
            gr.Markdown("## 3. 获取推荐")
            style_input = gr.Textbox(label="想要的风格")
            temperature_input = gr.Number(label="当前温度（可选）")
            mood_input = gr.Textbox(label="当前心情（可选）")
            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=os.getenv('API_HOST', 'github'),
                label="选择模型"
            )
            recommend_button = gr.Button("获取推荐")
            
            # 显示推荐结果
            gr.Markdown("## 4. 推荐搭配")
            recommendation_gallery = gr.Gallery(label="推荐搭配", show_label=False)
            recommendation_text = gr.Textbox(label="推荐说明", interactive=False, lines=10)
    
    # 绑定事件
    start_button.click(    
        start_crawler,
        outputs=login_status
    )
    
    process_button.click(
        process_data,
        outputs=gallery
    )
    
    model_dropdown.change(
        update_model,
        inputs=model_dropdown,
    )
    recommend_button.click(
        get_recommendation,
        inputs=[style_input, temperature_input, mood_input],
        outputs=[recommendation_gallery, recommendation_text]
    )

if __name__ == "__main__":
    print(f"Using API_HOST: {os.getenv('API_HOST', 'github')}")
    print(f"Using GITHUB_MODEL: {os.getenv('GITHUB_MODEL', 'gpt-4o')}")
    # 设置代理
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:2802"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2802"

    # # 跳过本地地址的代理
    # os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    demo.launch(
        server_name="127.0.0.1",  # 使用本地地址
        # server_port=7860,         # 指定端口
        share=False,              # 不创建公共链接
        show_error=True           # 显示错误信息
    ) 