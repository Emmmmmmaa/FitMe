# from typing import Dict, List, Tuple
# import numpy as np
# from dataclasses import dataclass
# from enum import Enum

# class Style(Enum):
#     SPORTY = "运动风"
#     CASUAL = "途行风"
#     FRENCH = "法式复古"
#     STREET = "街头辣妹"
#     FORMAL = "宴会/静态"

# @dataclass
# class ClothingItem:
#     type: str
#     colors: List[Tuple[int, int, int]]
#     exposure: float
#     style: Style
#     image_path: str

# class FashionRulesEngine:
#     def __init__(self):
#         self.color_compatibility = {
#             # 相似色
#             "similar": 0.8,
#             # 互补色
#             "complementary": 0.6,
#             # 对比色
#             "contrast": 0.4
#         }
        
#         self.style_compatibility = {
#             Style.SPORTY: [Style.CASUAL, Style.STREET],
#             Style.CASUAL: [Style.SPORTY, Style.FRENCH],
#             Style.FRENCH: [Style.CASUAL, Style.FORMAL],
#             Style.STREET: [Style.SPORTY, Style.CASUAL],
#             Style.FORMAL: [Style.FRENCH]
#         }
    
#     def calculate_color_match(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
#         """计算两个颜色的匹配度"""
#         # 转换为HSV颜色空间
#         hsv1 = self._rgb_to_hsv(color1)
#         hsv2 = self._rgb_to_hsv(color2)
        
#         # 计算色相差异
#         hue_diff = min(abs(hsv1[0] - hsv2[0]), 360 - abs(hsv1[0] - hsv2[0]))
        
#         if hue_diff < 30:  # 相似色
#             return self.color_compatibility["similar"]
#         elif 150 < hue_diff < 210:  # 互补色
#             return self.color_compatibility["complementary"]
#         else:  # 对比色
#             return self.color_compatibility["contrast"]
    
#     def calculate_style_match(self, style1: Style, style2: Style) -> float:
#         """计算两个风格的匹配度"""
#         if style1 == style2:
#             return 1.0
#         elif style2 in self.style_compatibility[style1]:
#             return 0.7
#         return 0.3
    
#     def calculate_exposure_balance(self, top: ClothingItem, bottom: ClothingItem) -> float:
#         """计算上下装露肤度的平衡度"""
#         exposure_diff = abs(top.exposure - bottom.exposure)
#         if exposure_diff < 2:
#             return 1.0
#         elif exposure_diff < 4:
#             return 0.7
#         return 0.3
    
#     def recommend_outfit(self, 
#                         tops: List[ClothingItem], 
#                         bottoms: List[ClothingItem], 
#                         target_style: Style,
#                         temperature: float = None) -> List[Tuple[ClothingItem, ClothingItem, float]]:
#         """推荐搭配"""
#         recommendations = []
        
#         for top in tops:
#             for bottom in bottoms:
#                 # 计算颜色匹配度
#                 color_score = max(
#                     self.calculate_color_match(top_color, bottom_color)
#                     for top_color in top.colors
#                     for bottom_color in bottom.colors
#                 )
                
#                 # 计算风格匹配度
#                 style_score = self.calculate_style_match(top.style, target_style) * \
#                             self.calculate_style_match(bottom.style, target_style)
                
#                 # 计算露肤度平衡
#                 exposure_score = self.calculate_exposure_balance(top, bottom)
                
#                 # 温度影响（可选）
#                 temp_score = 1.0
#                 if temperature is not None:
#                     if temperature > 25:  # 热天
#                         temp_score = 0.5 if top.exposure < 3 else 1.0
#                     else:  # 冷天
#                         temp_score = 0.5 if top.exposure > 7 else 1.0
                
#                 # 综合评分
#                 total_score = (color_score * 0.4 + 
#                              style_score * 0.3 + 
#                              exposure_score * 0.2 + 
#                              temp_score * 0.1)
                
#                 recommendations.append((top, bottom, total_score))
        
#         # 按评分排序
#         recommendations.sort(key=lambda x: x[2], reverse=True)
#         return recommendations
    
#     def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
#         """RGB转HSV"""
#         r, g, b = [x/255.0 for x in rgb]
#         max_val = max(r, g, b)
#         min_val = min(r, g, b)
#         diff = max_val - min_val
        
#         if max_val == min_val:
#             h = 0
#         elif max_val == r:
#             h = (60 * ((g - b) / diff) + 360) % 360
#         elif max_val == g:
#             h = (60 * ((b - r) / diff) + 120) % 360
#         else:
#             h = (60 * ((r - g) / diff) + 240) % 360
            
#         s = 0 if max_val == 0 else diff / max_val
#         v = max_val
        
#         return (h, s, v) 