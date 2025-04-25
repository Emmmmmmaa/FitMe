# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import torch
# from PIL import Image
# import numpy as np
# from typing import Dict, List, Tuple
# import cv2

# #

# class ClothingAnalyzer:
#     def __init__(self):
#         # 初始化Segformer模型
#         self.processor = AutoImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
#         self.model = AutoModelForImageClassification.from_pretrained("mattmdjaga/segformer_b2_clothes")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
    # def analyze_image(self, image_path: str) -> Dict:
    #     """分析单张服装图片"""
    #     try:
    #         # 加载图片
    #         image = Image.open(image_path)
            
    #         # 预处理
    #         inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
    #         # 预测
    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
    #             predictions = outputs.logits.argmax(dim=1)
            
    #         # 获取主要颜色
    #         main_colors = self._extract_main_colors(image)
            
    #         # 分析露肤度
    #         exposure_score = self._analyze_exposure(image)
            
    #         return {
    #             "clothing_type": self._get_clothing_type(predictions),
    #             "main_colors": main_colors,
    #             "exposure_score": exposure_score,
    #             "style_suggestions": self._suggest_styles(main_colors, exposure_score)
    #         }
    #     except Exception as e:
    #         print(f"Error analyzing image {image_path}: {str(e)}")
    #         return {}
    
    # def _extract_main_colors(self, image: Image.Image, num_colors: int = 3) -> List[Tuple[int, int, int]]:
    #     """提取图片中的主要颜色"""
    #     # 将图片转换为numpy数组
    #     img_array = np.array(image)
        
    #     # 使用K-means聚类提取主要颜色
    #     pixels = img_array.reshape(-1, 3)
    #     pixels = np.float32(pixels)
        
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    #     _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
    #     # 转换回整数RGB值
    #     centers = np.uint8(centers)
    #     return [tuple(color) for color in centers]
    
    # def _analyze_exposure(self, image: Image.Image) -> float:
        # """分析图片的露肤度"""
        # # TODO: 实现更复杂的露肤度分析算法
        # # 这里使用简单的启发式方法
        # img_array = np.array(image)
        # skin_pixels = np.sum((img_array > 200) & (img_array < 240))
        # total_pixels = img_array.size
        # return (skin_pixels / total_pixels) * 10  # 0-10的评分
    
    # def _get_clothing_type(self, predictions: torch.Tensor) -> str:
    #     """根据模型预测获取服装类型"""
    #     # TODO: 实现更详细的服装类型映射
    #     clothing_types = {
    #         0: "Upper-clothes",
    #         1: "Skirt",
    #         2: "Pants",
    #         3: "Dress",
    #         4: "Outerwear"
    #     }
    #     return clothing_types.get(predictions.item(), "Unknown")
    
    # def _suggest_styles(self, colors: List[Tuple[int, int, int]], exposure: float) -> List[str]:
        # """根据颜色和露肤度建议可能的风格"""
        # styles = []
        
        # # 根据颜色饱和度判断
        # for color in colors:
        #     r, g, b = color
        #     saturation = max(r, g, b) - min(r, g, b)
        #     if saturation > 200:
        #         styles.append("街头辣妹")
        #     elif saturation < 100:
        #         styles.append("法式复古")
        
        # # 根据露肤度判断
        # if exposure > 5:
        #     styles.append("运动风")
        # elif exposure < 2:
        #     styles.append("宴会/静态")
        
        # return list(set(styles))  # 去重 