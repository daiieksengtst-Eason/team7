# ==================== 晚餐照片健康分析系統 - 高精準度版本 ====================
# 檔案名稱：app_team7_advanced.py

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import threading
from datetime import datetime
import json
import gc

print("="*50)
print("🍽️  晚餐照片健康分析系統 (高精準度版本)")
print("="*50)

# ==================== 擴充的健康評分字典 ====================
HEALTH_SCORES = {
    # 健康食物（加分）
    'apple': 20, 'banana': 18, 'orange': 22, 'broccoli': 25,
    'carrot': 20, 'chicken': 8, 'fish': 15, 'salad': 25,
    'tomato': 15, 'cucumber': 12, 'egg': 10, 'rice': 5,
    'sushi': 10, 'strawberry': 20, 'grape': 15, 'watermelon': 15,
    'spinach': 25, 'celery': 18, 'zucchini': 15, 'asparagus': 20,
    'green_beans': 18, 'cauliflower': 22, 'mushroom': 12,
    
    # 不健康食物（扣分）
    'pizza': -15, 'burger': -20, 'donut': -25, 'hot dog': -18,
    'sandwich': -5, 'french fries': -25, 'cake': -20, 'ice cream': -20,
    'pasta': -8, 'fried_chicken': -25, 'fried_fish': -20,
    'nuggets': -22, 'onion_rings': -20, 'fried_shrimp': -22,
    'cheeseburger': -22, 'bacon': -18, 'sausage': -15,
    'nachos': -15, 'chips': -18, 'croissant': -15,
    'doughnut': -25, 'cookie': -15, 'brownie': -18,
    'milkshake': -15, 'soda': -10, 'energy_drink': -12,
    
    # 中性食物
    'bread': 0, 'toast': 0, 'waffle': -5, 'pancake': -5,
}

# 油炸食品關鍵詞
FRIED_KEYWORDS = ['fried', 'fries', 'nuggets', 'onion_rings', 'croissant', 
                  'doughnut', 'donut', 'fried_chicken', 'fried_fish', 'fried_shrimp']

UNHEALTHY_KEYWORDS = ['burger', 'pizza', 'hot dog', 'cake', 'ice cream', 
                      'soda', 'chips', 'nachos', 'bacon', 'sausage']

# 設定暫存目錄
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==================== 全域模型（避免重複載入）====================
global_model = None

def get_model():
    """取得全域模型（單例模式）"""
    global global_model
    if global_model is None:
        print("🔄 載入 YOLOv8 模型中...")
        global_model = YOLO('yolov8n.pt')
        global_model.overrides['conf'] = 0.35
        global_model.overrides['iou'] = 0.45
        print("✅ 模型載入完成！")
    return global_model

# ==================== 分析核心功能（獨立函數）====================
def check_if_fried(food_name):
    """檢查食物是否為油炸食品"""
    food_lower = food_name.lower()
    
    for keyword in FRIED_KEYWORDS:
        if keyword in food_lower:
            return True, f"偵測到油炸食品 ({keyword})"
    
    oily_indicators = ['oil', 'fried', 'crispy', 'battered', 'tempura']
    for indicator in oily_indicators:
        if indicator in food_lower:
            return True, f"可能為油炸食品 ({indicator})"
    
    return False, ""

def analyze_with_verification(image_path, model):
    """進行雙重驗證的分析（獨立函數）"""
    results = model(image_path)
    
    health_score = 100
    detected_foods = []
    warnings = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[class_id]
                
                if conf > 0.35:
                    food_keywords = ['apple', 'banana', 'orange', 'pizza', 'burger', 
                                    'cake', 'carrot', 'broccoli', 'fish', 'chicken']
                    
                    is_food = any(keyword in name.lower() for keyword in food_keywords)
                    
                    if is_food or name in HEALTH_SCORES:
                        base_score = HEALTH_SCORES.get(name, 0)
                        
                        is_fried, reason = check_if_fried(name)
                        if is_fried:
                            base_score = -25
                            warnings.append(f"⚠️ {name}: {reason}")
                        
                        health_score += base_score
                        
                        detected_foods.append({
                            'name': name,
                            'confidence': conf,
                            'score': base_score,
                            'is_fried': is_fried
                        })
    
    final_score = max(0, min(100, health_score))
    return detected_foods, final_score, warnings

def get_detailed_recommendation(detected_foods, health_score):
    """根據偵測到的食物提供詳細建議"""
    recommendations = []
    food_names = [f['name'] if isinstance(f, dict) else f for f in detected_foods]
    
    has_vegetables = any(food in ['broccoli', 'carrot', 'salad', 'tomato', 
                                  'cucumber', 'spinach', 'celery', 'cauliflower'] 
                        for food in food_names)
    
    has_fried = any(check_if_fried(food)[0] for food in food_names)
    
    has_sugar = any(food in ['cake', 'donut', 'ice cream', 'cookie', 'brownie'] 
                   for food in food_names)
    
    if not has_vegetables:
        recommendations.append("🥬 建議增加蔬菜攝取量")
    
    if has_fried:
        recommendations.append("🍟 建議減少油炸食品，改為蒸、煮、烤的烹調方式")
    
    if has_sugar:
        recommendations.append("🍰 甜點熱量較高，建議適量食用")
    
    if health_score >= 80:
        recommendations.append("🎉 很棒！繼續保持均衡飲食")
    elif health_score >= 60:
        recommendations.append("👍 不錯，可以再多注意食物選擇")
    elif health_score >= 40:
        recommendations.append("📝 建議記錄飲食日記，找出改善空間")
    else:
        recommendations.append("⚠️ 建議諮詢營養師，調整飲食習慣")
    
    return recommendations

# ==================== GUI 應用 ====================
class AdvancedFoodAnalyzerGUI:
    def __init__(self):
        self.model = get_model()
        self.result_windows = []  # 追蹤開啟的結果視窗
        
        self.window = tk.Tk()
        self.window.title("晚餐照片健康分析 - 高精準度版本")
        self.window.geometry("900x700")
        self.window.configure(bg='#f0f0f0')
        
        # 設定視窗關閉事件
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.create_menu()
    
    def on_closing(self):
        """關閉視窗時清理資源"""
        for win in self.result_windows:
            try:
                win.destroy()
            except:
                pass
        self.window.destroy()
        gc.collect()
    
    def create_menu(self):
        """建立主選單"""
        for widget in self.window.winfo_children():
            widget.destroy()
        
        title = tk.Label(self.window, 
                        text="🍽️ 晚餐照片健康分析系統\n(高精準度版本)",
                        font=("微軟正黑體", 24, "bold"),
                        bg='#f0f0f0', fg='#333333')
        title.pack(pady=30)
        
        features_frame = tk.Frame(self.window, bg='#f0f0f0')
        features_frame.pack(pady=20)
        
        features = [
            "✅ 更精確的食物辨識",
            "🍟 油炸食品自動偵測",
            "📊 詳細健康評分與建議",
            "🎯 雙重驗證機制"
        ]
        
        for feature in features:
            lbl = tk.Label(features_frame, text=feature, 
                          font=("微軟正黑體", 11),
                          bg='#f0f0f0', fg='#666666')
            lbl.pack(anchor='w', pady=2)
        
        upload_btn = tk.Button(self.window,
                              text="📷 選擇照片進行分析",
                              command=self.upload_photo,
                              font=("微軟正黑體", 14),
                              bg="#4CAF50", fg="white",
                              width=20, height=2,
                              relief='flat', cursor='hand2')
        upload_btn.pack(pady=30)
        
        exit_btn = tk.Button(self.window,
                            text="❌ 退出",
                            command=self.on_closing,
                            font=("微軟正黑體", 12),
                            bg="#f44336", fg="white",
                            width=10, height=1,
                            relief='flat', cursor='hand2')
        exit_btn.pack(pady=10)
    
    def upload_photo(self):
        """上傳照片"""
        file_path = filedialog.askopenfilename(
            title="選擇晚餐照片",
            filetypes=[("圖片文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # 檢查檔案大小
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                messagebox.showwarning("檔案過大", "圖片檔案過大（超過10MB），可能會影響分析速度")
            self.analyze_food(file_path)
    
    def analyze_food(self, image_path):
        """分析食物並顯示結果"""
        result_window = tk.Toplevel(self.window)
        result_window.title("分析結果")
        result_window.geometry("1000x750")
        result_window.configure(bg='white')
        
        # 追蹤視窗
        self.result_windows.append(result_window)
        result_window.protocol("WM_DELETE_WINDOW", lambda: self.close_result_window(result_window))
        
        # 顯示照片
        img = Image.open(image_path)
        img.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(img)
        
        img_label = tk.Label(result_window, image=photo, bg='white')
        img_label.image = photo
        img_label.pack(pady=20)
        
        result_text = scrolledtext.ScrolledText(result_window,
                                               width=70, height=25,
                                               font=("微軟正黑體", 10))
        result_text.pack(pady=10, padx=20, fill='both', expand=True)
        
        # 顯示分析中訊息
        result_text.insert(tk.END, "🔍 正在分析中，請稍候...\n", "title")
        result_text.update()
        
        # 執行分析（在背景執行緒避免 GUI 凍結）
        def do_analysis():
            try:
                detected_foods, final_score, warnings = analyze_with_verification(image_path, self.model)
                
                # 更新 UI（需要在主執行緒）
                result_window.after(0, lambda: self.display_results(
                    result_text, detected_foods, final_score, warnings
                ))
            except Exception as e:
                result_window.after(0, lambda: self.display_error(result_text, str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
        
        # 返回按鈕
        back_btn = tk.Button(result_window,
                            text="返回主選單",
                            command=lambda: self.close_result_window(result_window),
                            font=("微軟正黑體", 12),
                            bg="#2196F3", fg="white")
        back_btn.pack(pady=10)
    
    def display_results(self, result_text, detected_foods, final_score, warnings):
        """顯示分析結果"""
        result_text.delete(1.0, tk.END)
        
        result_text.insert(tk.END, "🔍 分析結果：\n", "title")
        result_text.insert(tk.END, "="*50 + "\n\n", "separator")
        
        if detected_foods:
            for food in detected_foods:
                score = food['score']
                if score > 0:
                    emoji = "✅"
                    effect = f"+{score}"
                elif score < 0:
                    emoji = "❌"
                    effect = f"{score}"
                else:
                    emoji = "•"
                    effect = "0"
                
                result_text.insert(tk.END, f"{emoji} {food['name']}\n", "food")
                result_text.insert(tk.END, f"   信心指數：{food['confidence']:.2%}\n", "detail")
                result_text.insert(tk.END, f"   健康影響：{effect}\n", "detail")
                
                if food['is_fried']:
                    result_text.insert(tk.END, f"   ⚠️ 油炸食品偵測！\n", "warning")
                
                result_text.insert(tk.END, "\n")
        else:
            result_text.insert(tk.END, "❌ 沒有偵測到食物\n\n", "error")
            result_text.insert(tk.END, "建議：\n", "tip")
            result_text.insert(tk.END, "• 確保照片清晰\n", "tip")
            result_text.insert(tk.END, "• 光線要充足\n", "tip")
            result_text.insert(tk.END, "• 食物要明顯可見\n", "tip")
        
        if warnings:
            result_text.insert(tk.END, "⚠️ 特殊警告：\n", "warning_title")
            for warning in warnings:
                result_text.insert(tk.END, f"   {warning}\n", "warning")
            result_text.insert(tk.END, "\n")
        
        result_text.insert(tk.END, "="*50 + "\n", "separator")
        result_text.insert(tk.END, f"📊 健康指數：{final_score}/100\n", "score")
        
        if final_score >= 80:
            result_text.insert(tk.END, "🥇 優良！飲食習慣很好！\n", "excellent")
        elif final_score >= 60:
            result_text.insert(tk.END, "🥈 良好，還可以再進步\n", "good")
        elif final_score >= 40:
            result_text.insert(tk.END, "🥉 普通，需要注意飲食選擇\n", "average")
        else:
            result_text.insert(tk.END, "⚠️ 需改善，建議調整飲食內容\n", "poor")
        
        result_text.insert(tk.END, "\n💡 改善建議：\n", "tip_title")
        recommendations = get_detailed_recommendation(detected_foods, final_score)
        
        for rec in recommendations:
            result_text.insert(tk.END, f"   {rec}\n", "tip")
        
        self.configure_text_tags(result_text)
    
    def display_error(self, result_text, error_msg):
        """顯示錯誤訊息"""
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "❌ 分析失敗\n\n", "error")
        result_text.insert(tk.END, f"錯誤訊息：{error_msg}\n", "detail")
    
    def close_result_window(self, window):
        """關閉結果視窗"""
        if window in self.result_windows:
            self.result_windows.remove(window)
        window.destroy()
        gc.collect()
    
    def configure_text_tags(self, text_widget):
        """設定文字樣式"""
        text_widget.tag_config("title", font=("微軟正黑體", 14, "bold"), foreground="#333333")
        text_widget.tag_config("separator", font=("微軟正黑體", 10), foreground="#999999")
        text_widget.tag_config("food", font=("微軟正黑體", 11, "bold"), foreground="#2E7D32")
        text_widget.tag_config("detail", font=("微軟正黑體", 9), foreground="#666666")
        text_widget.tag_config("warning", font=("微軟正黑體", 9, "bold"), foreground="#FF6F00")
        text_widget.tag_config("warning_title", font=("微軟正黑體", 11, "bold"), foreground="#D84315")
        text_widget.tag_config("score", font=("微軟正黑體", 12, "bold"), foreground="#1565C0")
        text_widget.tag_config("excellent", font=("微軟正黑體", 10, "bold"), foreground="#2E7D32")
        text_widget.tag_config("good", font=("微軟正黑體", 10, "bold"), foreground="#FF8F00")
        text_widget.tag_config("average", font=("微軟正黑體", 10), foreground="#C62828")
        text_widget.tag_config("poor", font=("微軟正黑體", 11, "bold"), foreground="#C62828")
        text_widget.tag_config("tip_title", font=("微軟正黑體", 11, "bold"), foreground="#1565C0")
        text_widget.tag_config("tip", font=("微軟正黑體", 10), foreground="#424242")
        text_widget.tag_config("error", font=("微軟正黑體", 11, "bold"), foreground="#C62828")
    
    def run(self):
        self.window.mainloop()

# ==================== FastAPI 應用（修正版）====================
fastapi_app = FastAPI(
    title="晚餐照片健康分析 API (高精準度)",
    description="分析食物照片並給出健康評分，具備油炸食品偵測功能",
    version="2.0.0"
)

# 啟用 CORS（允許跨域請求）
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetailedAnalysisResponse(BaseModel):
    success: bool
    detected_foods: List[Dict[str, Any]]
    health_score: int
    warnings: List[str]
    recommendations: List[str]
    analysis_time: str

@fastapi_app.post("/analyze_advanced", response_model=DetailedAnalysisResponse)
async def analyze_advanced(file: UploadFile = File(...)):
    """進階分析端點 - 包含油炸食品偵測"""
    temp_path = None
    try:
        # 檢查檔案類型
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/bmp"]:
            raise HTTPException(status_code=400, detail="只支援圖片檔案")
        
        # 檢查檔案大小（限制 10MB）
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="圖片檔案過大（最大 10MB）")
        
        image = Image.open(BytesIO(contents))
        
        # 使用唯一檔名避免衝突
        unique_id = str(uuid.uuid4())
        temp_path = os.path.join(TEMP_DIR, f"temp_{unique_id}.jpg")
        image.save(temp_path)
        
        # 使用全域模型進行分析
        model = get_model()
        detected_foods, health_score, warnings = analyze_with_verification(temp_path, model)
        
        # 產生建議
        recommendations = get_detailed_recommendation(detected_foods, health_score)
        
        return DetailedAnalysisResponse(
            success=True,
            detected_foods=detected_foods,
            health_score=health_score,
            warnings=warnings,
            recommendations=recommendations,
            analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失敗：{str(e)}")
    
    finally:
        # 清理暫存檔
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@fastapi_app.get("/")
async def root():
    """健康檢查"""
    return {
        "status": "online",
        "version": "2.0.0",
        "model_loaded": global_model is not None,
        "max_file_size_mb": 10,
        "features": ["油炸食品偵測", "詳細健康建議", "高精準度辨識"]
    }

@fastapi_app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "model_loaded": global_model is not None}

# ==================== 啟動 API 伺服器 ====================
def start_api_server():
    """在背景執行緒啟動 FastAPI 伺服器"""
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")

# ==================== 主程式 ====================
if __name__ == "__main__":
    print("\n🎯 高精準度版本特色：")
    print("• 油炸食品自動偵測與扣分")
    print("• 更詳細的健康建議")
    print("• 雙重驗證機制")
    print("• 信心門檻優化")
    print("• API 伺服器支援（含 CORS）")
    print("• 檔案大小限制 10MB\n")
    
    # 預先載入模型
    get_model()
    
    # 啟動 API 伺服器（背景執行緒）
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    print("🚀 API 伺服器已啟動在 http://localhost:8000")
    print("📖 API 文檔：http://localhost:8000/docs")
    print("🖼️  使用 POST /analyze_advanced 上傳圖片進行分析")
    print("🔧 健康檢查：http://localhost:8000/health\n")
    
    # 啟動 GUI
    gui = AdvancedFoodAnalyzerGUI()
    gui.run()