import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

# from src.utils.data_processor import DataProcessor
# from src.models.clothing_analyzer import ClothingAnalyzer
# from src.rules.fashion_rules import FashionRulesEngine, Style, ClothingItem
# from src.agents.fashion_agent import FashionAgent
from src.utils.taobao_crawler import TaobaoCrawler

def test_taobao_crawler():
    print("\n=== Testing Taobao Crawler ===")
    # 从环境变量获取cookie
    cookie = os.getenv("TAOBAO_COOKIE")
    if not cookie:
        print("TAOBAO_COOKIE not found in environment variables")
        return
        
    crawler = TaobaoCrawler(cookie)
    
    # 测试获取购买记录
    print("开始获取淘宝购买记录...")
    items = crawler.get_purchase_history(days=30)
    print(f"获取到 {len(items)} 件商品")
    
    if items:
        # 测试保存CSV
        print("\n保存数据到CSV...")
        crawler.save_to_csv(items, "data/test_taobao_purchases.csv")
        
        # 测试下载图片
        print("\n开始下载商品图片...")
        crawler.download_images(items, "data/test_images")
        print("图片下载完成")
    else:
        print("\n没有获取到商品数据，可能的原因：")
        print("1. Cookie可能已过期")
        print("2. 指定时间范围内没有购买记录")
        print("3. 网页结构可能发生变化")

# def test_data_processor():
#     print("\n=== Testing Data Processor ===")
#     processor = DataProcessor("data")
    
#     # 创建测试数据
#     test_csv = "data/test_data.csv"
#     test_image_url = "https://example.com/test.jpg"  # 替换为实际的测试图片URL
    
#     # 测试加载CSV
#     df = processor.load_taobao_data(test_csv)
#     print(f"Loaded CSV with {len(df)} rows")
    
#     # 测试下载图片
#     processor.download_images([test_image_url], "data/test_images")
#     print("Downloaded test image")
    
#     # 测试元数据保存和加载
#     test_metadata = {"test": "data"}
#     processor.save_metadata(test_metadata)
#     loaded_metadata = processor.load_metadata()
#     print(f"Metadata test: {loaded_metadata == test_metadata}")

# def test_clothing_analyzer():
#     print("\n=== Testing Clothing Analyzer ===")
#     analyzer = ClothingAnalyzer()
    
#     # 使用测试图片
#     test_image_path = "data/test_images/test.jpg"
#     if os.path.exists(test_image_path):
#         analysis = analyzer.analyze_image(test_image_path)
#         print("Analysis results:")
#         print(f"Clothing type: {analysis.get('clothing_type')}")
#         print(f"Main colors: {analysis.get('main_colors')}")
#         print(f"Exposure score: {analysis.get('exposure_score')}")
#         print(f"Style suggestions: {analysis.get('style_suggestions')}")
#     else:
#         print("Test image not found")

# def test_fashion_rules():
#     print("\n=== Testing Fashion Rules Engine ===")
#     rules_engine = FashionRulesEngine()
    
#     # 创建测试服装项
#     test_top = ClothingItem(
#         type="Upper-clothes",
#         colors=[(255, 0, 0), (200, 0, 0)],  # 红色系
#         exposure=5.0,
#         style=Style.SPORTY,
#         image_path="test_top.jpg"
#     )
    
#     test_bottom = ClothingItem(
#         type="Pants",
#         colors=[(0, 0, 255), (0, 0, 200)],  # 蓝色系
#         exposure=2.0,
#         style=Style.SPORTY,
#         image_path="test_bottom.jpg"
#     )
    
#     # 测试颜色匹配
#     color_score = rules_engine.calculate_color_match(
#         test_top.colors[0], test_bottom.colors[0]
#     )
#     print(f"Color match score: {color_score}")
    
#     # 测试风格匹配
#     style_score = rules_engine.calculate_style_match(
#         test_top.style, test_bottom.style
#     )
#     print(f"Style match score: {style_score}")
    
#     # 测试露肤度平衡
#     exposure_score = rules_engine.calculate_exposure_balance(test_top, test_bottom)
#     print(f"Exposure balance score: {exposure_score}")
    
#     # 测试完整推荐
#     recommendations = rules_engine.recommend_outfit(
#         [test_top], [test_bottom], Style.SPORTY, 20.0
#     )
#     print(f"Number of recommendations: {len(recommendations)}")
#     if recommendations:
#         print(f"Best match score: {recommendations[0][2]}")

# def test_fashion_agent():
    # print("\n=== Testing Fashion Agent ===")
    # agent = FashionAgent()
    
    # # 测试服装分析
    # test_image_path = "data/test_images/test.jpg"
    # if os.path.exists(test_image_path):
    #     clothing_features = agent.analyze_clothing(test_image_path, {})
    #     print("Clothing features:", clothing_features)
        
    #     # 测试风格匹配
    #     matches = agent.match_style(clothing_features, "SPORTY")
    #     print("Style matches:", matches)
        
    #     # 测试推荐生成
    #     recommendation = agent.generate_recommendation(matches, {
    #         "temperature": 20.0,
    #         "mood": "元气"
    #     })
    #     print("Recommendation:", recommendation)
    # else:
    #     print("Test image not found")

def main():
    # 创建测试目录
    os.makedirs("data/test_images", exist_ok=True)
    
    # 运行测试
    test_taobao_crawler()  # 先测试爬虫
    # test_data_processor()
    # test_clothing_analyzer()
    # test_fashion_rules()
    # test_fashion_agent()

if __name__ == "__main__":
    main() 