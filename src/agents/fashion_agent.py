import autogen
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import pandas as pd
from dotenv import load_dotenv
import azure.identity
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

load_dotenv(override=True)

class FashionAgent:
    def __init__(self, clothing_data: Union[pd.DataFrame, str]):
        # 加载衣物数据
        if isinstance(clothing_data, str):
            # 检查文件是否存在
            if not os.path.exists(clothing_data):
                raise FileNotFoundError(f"CSV文件不存在: {clothing_data}")
            
            # 读取CSV文件
            self.clothing_data = pd.read_csv(clothing_data)
            print(f"成功加载数据，共 {len(self.clothing_data)} 条记录")
            print("数据预览:")
            print(self.clothing_data.head())
        else:
            self.clothing_data = clothing_data
        
        # 配置 LLM
        API_HOST = os.getenv("API_HOST", "github")
        print(API_HOST)
        if API_HOST == "github":
            self.client = OpenAIChatCompletionClient(
                model=os.getenv("GITHUB_MODEL", "gpt-4o"), 
                api_key=os.environ["GITHUB_TOKEN"],
                base_url="https://models.inference.ai.azure.com"
            )
        elif API_HOST == "azure":
            token_provider = azure.identity.get_bearer_token_provider(
                azure.identity.DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAIChatCompletionClient(
                model=os.environ["AZURE_OPENAI_CHAT_MODEL"],
                api_version=os.environ["AZURE_OPENAI_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_ad_token_provider=token_provider
        )
            
        # 创建智能体
        self.fashion_agent = AssistantAgent(
            "fashion_expert",
            model_client=self.client,
            system_message="""你是一个专业的时尚搭配专家。请严格遵循以下规则：

            1. 数据限制：
                - 选择一件上衣和一件下装进行搭配（非常重要！）
                - 只能从提供的服装数据中选择搭配
                - 不能推荐数据中不存在的服装
                - 不能虚构或修改服装的属性

            2. 搭配规则：
                - 颜色匹配：主色相近 > 互补 > 其他对比色（优先级：高）
                - 风格匹配：风格必须与目标风格场景一致或兼容，上下装风格需要协调，避免风格冲突的搭配（如运动风配正装）（优先级：中-高）

            3. 环境适应：
                - 根据温度选择合适材质的服装
                    - 上下装适宜的温度需要一致，不得出现上下装温差过大，例如绒衣搭配短裙
                    - 当温度 > 25°C：选择轻薄、透气材质（如棉、麻、雪纺）
                    - 当温度 < 15°C：选择保暖材质（如羊毛、毛呢、摇粒绒）
                    - 当温度在15-25°C之间：选择适中厚度的材质
                    
                - 根据心情调整色彩搭配
                    - 心情愉悦时：可以选择明亮、活泼的颜色
                    - 心情平静时：可以选择柔和、中性的颜色
                    - 心情沮丧时：可以选择深色、低饱和度的颜色

            4. 服装数据限制：
                - 只能从提供的服装数据中选择搭配
                - 不能推荐数据中不存在的服装
                - 不能虚构或修改服装的属性

            你的任务是：
            1. 从提供的服装数据中选择最合适的搭配
            2. 生成具体的搭配建议
            3. 提供详细的推荐理由

            输出格式要求（JSON格式）：
            {
                "top": {
                    "title": "上衣名称",
                    "image_url": "图片URL"
                },
                "bottom": {
                    "title": "下装名称",
                    "image_url": "图片URL"
                },
                "reason": "基于颜色和风格的推荐理由"
            }"""
        )
        '''
        其他规则(TBD)：
            - 露肤度平衡：上身为短款/露肩/抹胸时，推荐下身为长裤/高腰裙
        '''

    # def _filter_clothing(self, style_preference: str, temperature: Optional[float] = None) -> pd.DataFrame:
    #     """根据用户偏好筛选服装，没有办法完全字段相同进行匹配"""
    #     print(f"\n开始筛选服装，风格偏好: {style_preference}, 温度: {temperature}")
        
    #     # 1. 首先筛选出可用的服装
    #     available_clothing = self.clothing_data
        
    #     # 2. 根据风格偏好筛选
    #     if style_preference.lower() != 'unknown':
    #         style_matches = available_clothing[
    #             available_clothing['style'].str.contains(style_preference, case=False, na=False)
    #         ]
    #         print(f"风格匹配的服装数量: {len(style_matches)}")
    #     else:
    #         style_matches = available_clothing
        
    #     # 3. 如果温度信息存在，根据温度筛选合适的材质
    #     if temperature is not None:
    #         if temperature > 25:  # 高温
    #             style_matches = style_matches[
    #                 ~style_matches['title'].str.contains('羊毛|毛呢|厚|保暖', case=False, na=False)
    #             ]
    #         elif temperature < 15:  # 低温
    #             style_matches = style_matches[
    #                 style_matches['title'].str.contains('羊毛|毛呢|厚|保暖', case=False, na=False)
    #             ]
    #         print(f"温度筛选后的服装数量: {len(style_matches)}")
        
    #     return style_matches

    # def _format_clothing_data(self, clothing_df: pd.DataFrame) -> str:
    #     """格式化服装数据，使其更易读"""
    #     if clothing_df.empty:
    #         return "没有找到匹配的服装。"
            
    #     # 按类型分组
    #     grouped = clothing_df.groupby('type')
    #     formatted_data = []
        
    #     for type_name, group in grouped:
    #         formatted_data.append(f"\n{type_name.upper()}类服装:")
    #         for _, row in group.iterrows():
    #             formatted_data.append(
    #                 f"- {row['title']} (风格: {row['style']}, 颜色: {row['color']}, "
    #                 f"露肤度: {row['exposure_level']})"
    #             )
        
    #     return "\n".join(formatted_data)
    
    async def process_request(self, style_preference: str, 
                            temperature: Optional[float] = None, 
                            mood: Optional[str] = None) -> Dict:
        """处理完整的推荐请求"""
        
        # 格式化服装数据，确保URL完整
        formatted_data = self.clothing_data.to_dict('records')
        
        # 构建初始提示
        initial_prompt = f"""
        请帮我推荐服装搭配：
        
        1. 可用服装数据：
        {json.dumps(formatted_data, ensure_ascii=False, indent=2)}
        
        2. 用户偏好：
        - 目标风格：{style_preference}
        - 当前温度：{temperature if temperature else '未指定'}
        - 当前心情：{mood if mood else '未指定'}
        """
        
        # print("\n发送给模型的提示:")
        # print(initial_prompt)
        
        # 发送消息并获取响应
        response = await self.fashion_agent.on_messages(
            [TextMessage(content=initial_prompt, source="user")],
            cancellation_token=CancellationToken(),
        )
        
        return response.chat_message.content
    

async def main():
    try:
        # 创建FashionAgent实例，直接传入CSV文件路径
        agent = FashionAgent("data/new_processed_taobao_purchases.csv")
        
        # 获取用户偏好
        style_preference = input("请输入你想要的风格: ")
        temperature = input("请输入当前温度(可选，直接回车跳过): ")
        temperature = float(temperature) if temperature else None
        mood = input("请输入当前心情(可选，直接回车跳过): ")
        
        # 处理推荐请求
        result = await agent.process_request(
            style_preference=style_preference,
            temperature=temperature,
            mood=mood
        )
        
        # 打印结果
        print("\n推荐结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 