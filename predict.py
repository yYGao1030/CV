import torch
import cv2
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

class FruitDetectionSystem:
    def __init__(self, weights_path='best.pt', device='auto', img_size=640):
        """
        初始化水果检测系统
        
        参数:
            weights_path: 权重文件路径
            device: 计算设备 ('cuda', 'cpu' 或 'auto')
            img_size: 模型输入尺寸
        """
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件未找到: {self.weights_path}")

        self.device = self._select_device(device)
        self.img_size = img_size
        
        # 中文字体配置
        self.font_path = self._get_chinese_font()
        if not self.font_path:
            print("警告: 未找到中文字体，将使用默认字体")
        
        # 类别配置(RGB模型)
        self.class_config = {
            0: {'name': 'apple', 'color': (255, 0, 0), 'display': '苹果(红)'},
            1: {'name': 'banana', 'color': (255, 255, 0), 'display': '香蕉(黄)'}, 
            2: {'name': 'orange', 'color': (0, 255, 0), 'display': '橘子(橙)'}
        }
        
        self.model = self._load_model()
        self.model.to(self.device).eval()
        print(f"[系统初始化完成] 设备: {self.device}")

    def _select_device(self, device):
        """自动选择计算设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _get_chinese_font(self, font_size=20):
        """获取系统中文字体"""
        try:
            # 尝试查找常见中文字体
            for font in fm.findSystemFonts():
                if 'simhei' in font.lower() or 'msyh' in font.lower() or 'wqy' in font.lower():
                    return font
            return None
        except:
            return None

    def _load_model(self):
        """安全加载模型"""
        load_attempts = [
            self._load_from_local_repo,
            self._load_via_torch_hub,
            self._load_directly
        ]
        
        for attempt in load_attempts:
            try:
                model = attempt()
                print(f"[模型加载] 成功通过 {attempt.__name__}")
                return model
            except Exception as e:
                print(f"[模型加载] {attempt.__name__} 失败: {str(e)}")
                time.sleep(1)
        
        raise RuntimeError("所有模型加载方式均失败")

    def _load_from_local_repo(self):
        """从本地克隆的仓库加载"""
        local_repo = Path('./yolov5')
        if local_repo.exists():
            return torch.hub.load(str(local_repo), 'custom', 
                                path=str(self.weights_path), 
                                source='local')
        raise RuntimeError("本地YOLOv5仓库不存在")

    def _load_via_torch_hub(self):
        """通过torch.hub加载"""
        return torch.hub.load('ultralytics/yolov5', 'custom', 
                            path=str(self.weights_path))

    def _load_directly(self):
        """直接加载模型文件"""
        from models.experimental import attempt_load
        return attempt_load(str(self.weights_path))

    def detect(self, img_input, conf_threshold=0.3, iou_threshold=0.45, visualize=True):
        """
        执行水果检测
        
        参数:
            img_input: 图像路径或numpy数组
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            visualize: 是否返回可视化结果
            
        返回:
            dict: 检测结果
            ndarray: 可视化图像
        """
        img = self._load_image(img_input)
        if img is None:
            raise ValueError("无法加载输入图像")
        
        with torch.no_grad():
            results = self.model(img)
        
        counts, output_img = self._process_results(
            results, 
            img.copy() if visualize else None
        )
        
        result = {
            'counts': counts,
            'detections': results.pandas().xyxy[0].to_dict('records'),
            'image_size': img.shape[:2]
        }
        
        return (result, output_img) if visualize else result

    def _load_image(self, img_input):
        """加载图像"""
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif isinstance(img_input, np.ndarray):
            return img_input.copy()
        return None

    def _process_results(self, results, output_img=None):
        """处理检测结果"""
        detections = results.pandas().xyxy[0]
        counts = defaultdict(int)
        
        if output_img is not None:
            # 使用PIL绘制中文文本
            pil_img = Image.fromarray(output_img)
            draw = ImageDraw.Draw(pil_img)
            
            if self.font_path:
                font = ImageFont.truetype(self.font_path, 20)
            else:
                font = ImageFont.load_default()
            
            for _, det in detections.iterrows():
                class_id = int(det['class'])
                if class_id not in self.class_config:
                    continue
                    
                class_info = self.class_config[class_id]
                counts[class_info['display']] += 1
                
                # 绘制边界框
                x1, y1, x2, y2 = map(int, [
                    det['xmin'], det['ymin'],
                    det['xmax'], det['ymax']
                ])
                draw.rectangle([x1, y1, x2, y2], 
                              outline=class_info['color'], 
                              width=2)
                
                # 绘制中文标签
                label = f"{class_info['display']} {det['confidence']:.2f}"
                draw.text((x1, y1-25), label, 
                          fill=class_info['color'], 
                          font=font)
            
            # 添加统计信息
            stats = " | ".join([f"{k}:{v}" for k, v in counts.items()])
            draw.text((10, 10), stats, fill=(0, 0, 0), font=font)
            
            # 转换回OpenCV格式
            output_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            for _, det in detections.iterrows():
                class_id = int(det['class'])
                if class_id in self.class_config:
                    counts[self.class_config[class_id]['display']] += 1
        
        return dict(counts), output_img

    def detect_realtime(self, source=0, window_name="水果检测系统"):
        """实时检测"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result, output_frame = self.detect(frame)
                cv2.imshow(window_name, output_frame)
                
                stats = " | ".join([f"{k}:{v}" for k, v in result['counts'].items()])
                print(f"\r检测结果: {stats}", end="", flush=True)
                
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n[系统] 实时检测已停止")

if __name__ == "__main__":
    try:
        detector = FruitDetectionSystem(weights_path='best.pt')
        
        # 单张图像检测
        result, output_img = detector.detect('abo.jpg')
        print("\n检测结果:")
        for fruit, count in result['counts'].items():
            print(f"{fruit}: {count}个")
        
        cv2.imwrite('output.jpg', output_img)
        print("结果已保存到 output.jpg")
        
        # 实时检测 (取消注释运行)
        # print("\n启动实时检测... (按q退出)")
        # detector.detect_realtime()
        
    except Exception as e:
        print(f"[错误] {str(e)}")