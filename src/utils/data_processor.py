'''暂未剔除退款的记录'''



import pandas as pd
import requests
from PIL import Image
import os
from typing import List, Dict
import json
import re

class DataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
    def is_top(self, name: str) -> bool:
        """判断是否为上衣"""
        keywords = ['背心', '上衣', 'T恤', '抹胸', '吊带', '露脐', '短袖', '衬衫', '外套', '夹克', '卫衣']
        return any(kw in name for kw in keywords)
    
    def is_bottom(self, name: str) -> bool:
        """判断是否为下装"""
        keywords = ['短裤', '长裤', '裤子', '半身裙']
        return any(kw in name for kw in keywords)
    
    def is_dress(self, name: str) -> bool:
        """判断是否为连衣裙/连体裤"""
        keywords = ['连衣裙', '连体裤', '套装', '长裙', '吊带裙', '背带裤']
        return any(kw in name for kw in keywords)
    
    def is_accessory(self, name: str) -> bool:
        """判断是否为配饰"""
        keywords = ['帽子', '项链', '耳环', '手链', '戒指', '发饰', '围巾', '手套', '袜子', '包', '腰带', '眼镜', '口罩', '帽子', '鞋','袜子']
        return any(kw in name for kw in keywords)
    
    def estimate_exposure(self, name: str) -> str:
        """估算露肤度"""
        high = ['抹胸', '露脐', '吊带']
        medium = ['短袖', '背心', '短裙', '短裤']
        low = ['长裙', '长裤', '毛呢']
        
        if any(kw in name for kw in high):
            return 'high'
        elif any(kw in name for kw in medium):
            return 'medium'
        elif any(kw in name for kw in low):
            return 'low'
        return 'unknown'
    
    def extract_style(self, name: str, fallback: str = None) -> str:
        """提取风格关键词"""
        keywords = ['通勤', '辣妹', '运动', '学院', '复古', '法式']
        style_map = {
            '通勤': 'commuter',
            '辣妹': 'trendy',
            '运动': 'sports',
            '学院': 'academic',
            '复古': 'retro',
            '法式': 'french'
        }
        for kw in keywords:
            if kw in name:
                return kw #style_map[kw]
        return fallback if fallback else 'unknown'

    def extract_color_size(self, spec: str) -> tuple:
        """从规格中提取颜色和尺码"""
        color = ''
        size = ''
        
        if pd.isna(spec):
            return color, size
            
        # 尝试提取颜色
        color_section = re.search(r'(?:颜色分类|主要颜色)[:：]([^:：]+)', str(spec))
        if color_section:
            # 获取颜色部分的文本并清理
            color_text = color_section.group(1).strip()
            
            # 先尝试匹配 xxx色
            color_match = re.search(r'([^\s,，]+色)', color_text)
            if color_match:
                color = color_match.group(1)
            else:
                # 如果没找到xxx色，则保留第一段非空文本（处理类似"浆果玫红"这样的组合词）
                color = re.split(r'[,，\s\-]+', color_text)[0].strip()
            
        # 尝试提取尺码
        size_section = re.search(r'尺码[:：]([^:：]+)', str(spec))
        if size_section:
            size = size_section.group(1).strip()
            # 清理尺码中的特殊字符和乱码
            size = re.sub(r'[\[\]【】\(\)]', '', size)
            
        return color, size
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理淘宝购买记录数据"""
        # 清理标题中的[交易快照]
        df['title'] = df['title'].str.replace(r'\[交易快照\]$', '', regex=True).str.strip()
        
        # 清理 image_url 中的 _80x80.jpg
        df['image_url'] = df['image_url'].str.replace(r'_80x80\.jpg$', '_640x640.jpg', regex=True)

        # 添加必要的列
        if "type" not in df.columns:
            df["type"] = ""  # 服装类型
        if "style" not in df.columns:
            df["style"] = ""  # 风格
        if "exposure_level" not in df.columns:
            df["exposure_level"] = ""  # 露肤度
                   
        # 从specification提取颜色和尺码
        df[['color', 'size']] = pd.DataFrame(
            df['specification'].apply(self.extract_color_size).tolist(),
            index=df.index
        )
        
        # 添加新的处理逻辑
        df['type'] = df['title'].apply(lambda x: 
            '上衣' if self.is_top(x) 
            else ('下装' if self.is_bottom(x) 
            else ('连衣裙/裤' if self.is_dress(x)
            else ('配饰' if self.is_accessory(x)
            else '未知'))))
            
        df['exposure_level'] = df['title'].apply(self.estimate_exposure)
        df['style'] = df.apply(lambda row: self.extract_style(str(row['title'])), axis=1)
        
        # 判断是否是服饰（类型不为未知，且颜色和尺码都不为空）
        df['is_clothing'] = (df['type'].apply(lambda x: x != '未知') & 
                           df['color'].str.len().gt(0) & 
                           df['size'].str.len().gt(0))

        # 剔除退款的记录 ！！暂未剔除
        # df = df[~df['status'].str.contains('查看退款', na=False)]
        
        # 剔除非服装类商品
        df = df[df['is_clothing'] == True]
            
        return df
    
    # def download_images(self, image_urls: List[str], output_dir: str):
    #     """下载并保存图片"""
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     for url in image_urls:
    #         try:
    #             response = requests.get(url)
    #             if response.status_code == 200:
    #                 filename = os.path.join(output_dir, f"{hash(url)}.jpg")
    #                 with open(filename, "wb") as f:
    #                     f.write(response.content)
    #         except Exception as e:
    #             print(f"Error downloading {url}: {str(e)}")
    
    # def save_metadata(self, metadata: Dict):
    #     """保存元数据"""
    #     with open(self.metadata_file, "w", encoding="utf-8") as f:
    #         json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # def load_metadata(self) -> Dict:
    #     """加载元数据"""
    #     if os.path.exists(self.metadata_file):
    #         with open(self.metadata_file, "r", encoding="utf-8") as f:
    #             return json.load(f)
    #     return {}
    
    # def process_image(self, image_path: str) -> Dict:
    #     """处理单张图片，提取特征"""
    #     try:
    #         with Image.open(image_path) as img:
    #             # 这里可以添加图像处理逻辑
    #             # 例如：调整大小、格式转换等
    #             return {
    #                 "width": img.width,
    #                 "height": img.height,
    #                 "format": img.format,
    #                 "path": image_path
    #             }
    #     except Exception as e:
    #         print(f"Error processing image {image_path}: {str(e)}")
    #         return {} 


def main():
    # 初始化数据处理器
    processor = DataProcessor(data_dir="data")
    
    # 加载淘宝购买数据
    raw_data = pd.read_csv("data/taobao_purchases.csv")
    df = processor.process_data(raw_data)
    
    # 保存处理后的数据
    output_path = "data/processed_taobao_purchases.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    main() 