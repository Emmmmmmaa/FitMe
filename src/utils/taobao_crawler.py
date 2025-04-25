import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from typing import List, Dict
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class TaobaoCrawler:
    def __init__(self):
        """
        初始化爬虫
        """
        self.options = webdriver.ChromeOptions()
        # 添加一些选项来避免被检测
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        # 如果需要无头模式（不显示浏览器窗口）
        # self.options.add_argument('--headless')
        
        self.driver = None
        
    def login(self):
        """
        使用二维码登录淘宝，并导航到已买到的宝贝页面
        """
        if self.driver is None:
            self.driver = webdriver.Chrome(options=self.options)
            
        # 清除所有 cookies
        self.driver.delete_all_cookies()
        
        # 访问登录页面
        self.driver.get('https://login.taobao.com')

        try:
            # 等待登录成功，最多等待300秒
            WebDriverWait(self.driver, 300).until(
                lambda driver: 'login' not in driver.current_url
            )
            print("登录成功！")
            
            # 等待页面加载完成
            time.sleep(3)  # 给页面一些加载时间
            
            # 点击"已买到的宝贝"按钮
            try:
                bought_items_link = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, "已买到的宝贝"))
                )
                bought_items_link.click()
                print("成功导航到已买到的宝贝页面")
                return True
            except TimeoutException:
                print("无法找到或点击'已买到的宝贝'按钮")
                return False
                
        except TimeoutException:
            print("登录超时，请重试")
            return False
                
            
    def get_purchase_history(self, days: int = 30) -> List[Dict]:
        """
        获取指定天数内的购买记录
        :param days: 要获取的天数
        :return: 购买记录列表
        """
        if self.driver is None:
            print("请先调用 login() 方法完成登录")
            return []
            
        # 等待页面加载完成
        try:
            # 等待订单容器加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'index-mod__order-container___1ur4-'))
            )
        except TimeoutException:
            print("页面加载超时")
            return []
            
        # 获取订单数据
        orders = []
        try:
            # 获取所有订单容器
            order_containers = self.driver.find_elements(By.CLASS_NAME, 'index-mod__order-container___1ur4-')
            print(f"找到 {len(order_containers)} 个订单日期组")
            
            for container in order_containers:
                try:
                    # 获取每个订单项
                    order_items = container.find_elements(By.TAG_NAME, 'tbody')
                    
                    for item in order_items[1:]:  # 跳过第一个tbody（它是订单头部）
                        try:
                            # 获取商品标题和规格
                            title_element = item.find_element(By.CSS_SELECTOR, 'p[data-reactid*="0.0.1.0"]')
                            title = title_element.text.strip()
                            
                            # 获取规格信息
                            spec_element = item.find_element(By.CSS_SELECTOR, 'p[data-reactid*="0.0.1.1"]')
                            spec = spec_element.text.strip()

                            # 获取图片链接
                            img_element = item.find_element(By.CSS_SELECTOR, '.production-mod__pic___2Wuak img')
                            img_url = img_element.get_attribute('src')
                            
                            # 获取价格
                            price_element = item.find_element(By.CSS_SELECTOR, 'p[data-reactid*="1.0.1"]')
                            price = price_element.text.strip()
                            
                            # 获取订单状态 -- 后续提出交易关闭的部分
                            status_element = item.find_element(By.CSS_SELECTOR, 'td[class*="sol-mod__no-br"] + td + td + td')
                            status = status_element.text.strip()
                            
                            order_info = {
                                'title': title,
                                'specification': spec,
                                'image_url': img_url,  # 新增图片链接
                                'price': price,
                                'status': status
                            }
                            orders.append(order_info)
                            
                        except Exception as e:
                            print(f"解析订单项时出错: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"处理订单容器时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"获取订单列表时出错: {str(e)}")
            
        return orders
        
    def save_to_csv(self, items: List[Dict], output_path: str):
        """
        将商品信息保存为CSV文件
        """
        import pandas as pd
        df = pd.DataFrame(items)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存 {len(items)} 条记录到 {output_path}")
        
    def close(self):
        """
        关闭浏览器
        """
        if self.driver:
            self.driver.quit()
            self.driver = None

def main():
    crawler = TaobaoCrawler()
    
    # 登录
    if crawler.login():
        # 获取最近30天的购买记录
        items = crawler.get_purchase_history(days=30) # 目前未筛选，仅爬第一页；之后可以补充用日期筛选，爬取多页
        
        # 保存为CSV
        if items:
            crawler.save_to_csv(items, "data/taobao_purchases.csv")
    
    # 关闭浏览器
    crawler.close()

if __name__ == "__main__":
    main() 