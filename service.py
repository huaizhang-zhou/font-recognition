import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import base64
import pickle
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import cv2
import os
import sys


# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from font_recognition.SCAE import SCAE, Encoder
    from font_recognition.CNN import CNN
    from general_code.utils import load_image, img_to_tensor, image_sampling
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保项目结构正确，并安装了必要的依赖包")
    sys.exit(1)


class FontRecognitionService:
    def __init__(self, scae_model_path, cnn_model_path, font_dict_path, device=None):
        """初始化字体识别服务

        Args:
            scae_model_path: SCAE模型路径
            cnn_model_path: CNN模型路径
            font_dict_path: 字体字典路径
            device: 指定设备（cuda/cpu），None则自动选择
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        # 加载字体字典
        try:
            with open(font_dict_path, "r", encoding="utf-8") as f:
                self.font_dict = json.load(f)
            self.reverse_font_dict = {v: k for k, v in self.font_dict.items()}
            self.num_fonts = len(self.font_dict)
            print(f"加载字体字典成功，共 {self.num_fonts} 种字体")
        except Exception as e:
            print(f"加载字体字典失败: {e}")
            raise

        # 初始化模型
        self.scae_model = None
        self.cnn_model = None

        # 加载SCAE模型
        print(f"正在加载SCAE模型: {scae_model_path}")
        try:
            if os.path.exists(scae_model_path):
                # 直接加载整个模型对象
                self.scae_model = torch.load(
                    scae_model_path, map_location=self.device, weights_only=False
                )
                self.scae_model.to(self.device)
                self.scae_model.eval()
                print("SCAE模型加载成功")

                # 固定SCAE编码器参数（用于CNN的特征提取）
                for param in self.scae_model.encoder.parameters():
                    param.requires_grad = False
            else:
                print(f"SCAE模型文件不存在: {scae_model_path}")
                raise FileNotFoundError(f"SCAE模型文件不存在: {scae_model_path}")
        except Exception as e:
            print(f"SCAE模型加载失败: {e}")
            raise

        # 加载CNN模型
        print(f"正在加载CNN模型: {cnn_model_path}")
        try:
            if os.path.exists(cnn_model_path):
                # 方法1：尝试直接加载整个模型对象
                cnn_loaded = torch.load(
                    cnn_model_path, map_location=self.device, weights_only=False
                )

                # 检查加载的是模型对象还是state_dict
                if isinstance(cnn_loaded, nn.Module):
                    # 加载的是整个模型对象
                    print("检测到CNN模型文件为完整模型对象")
                    self.cnn_model = cnn_loaded
                elif isinstance(cnn_loaded, dict):
                    # 加载的是state_dict，需要使用SCAE的编码器初始化CNN
                    print("检测到CNN模型文件为state_dict")
                    self.cnn_model = CNN(self.scae_model.encoder, self.num_fonts)
                    self.cnn_model.load_state_dict(cnn_loaded)
                else:
                    raise ValueError(f"未知的模型文件格式: {type(cnn_loaded)}")

                # 将模型移动到设备
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                print("CNN模型加载成功")

                # 打印模型信息
                print(f"CNN模型结构:")
                print(f"  编码器部分: {self.cnn_model.Cu}")
                print(
                    f"  分类器部分参数数量: {sum(p.numel() for p in self.cnn_model.Cs.parameters())}"
                )

            else:
                print(f"CNN模型文件不存在: {cnn_model_path}")
                raise FileNotFoundError(f"CNN模型文件不存在: {cnn_model_path}")
        except Exception as e:
            print(f"CNN模型加载失败: {e}")

            # 尝试备用方法：使用SCAE的编码器重新构建CNN
            print("尝试备用方法：使用SCAE编码器重新构建CNN...")
            try:
                self.cnn_model = CNN(self.scae_model.encoder, self.num_fonts)
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                print("警告: 使用未训练的CNN模型（随机初始化分类器）")
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                raise

        print("字体识别服务初始化完成")

    def preprocess_image(self, image, target_size=105):
        """预处理图像，与训练时保持一致

        Args:
            image: PIL Image对象
            target_size: 目标尺寸

        Returns:
            预处理后的tensor
        """
        # 调整大小并转换为灰度图
        if image.width != target_size or image.height != target_size:
            image = resize_with_padding(image, target_size, target_size)

        # 转换为灰度图
        if image.mode != "L":
            image = image.convert("L")

        # 转换为tensor
        img_tensor = img_to_tensor(image)

        # 添加batch维度
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict_single_image(self, image):
        """预测单张图像的字体

        Args:
            image: PIL Image对象

        Returns:
            预测结果字典
        """
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                # 使用CNN进行预测
                output = self.cnn_model(input_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)

                # 获取预测结果
                font_index = predicted.item()
                font_confidence = confidence.item()
                font_name = self.reverse_font_dict.get(font_index, "UNKNOWN")

                # 获取top-3预测结果
                top3_probs, top3_indices = torch.topk(probs, 3)
                top3_predictions = []
                for i in range(3):
                    idx = top3_indices[0][i].item()
                    prob = top3_probs[0][i].item()
                    font_name_i = self.reverse_font_dict.get(idx, "UNKNOWN")
                    top3_predictions.append(
                        {"font": font_name_i, "index": idx, "confidence": prob}
                    )

                result = {
                    "font": font_name,
                    "font_confidence": font_confidence,
                    "font_index": font_index,
                    "confidence": font_confidence,
                    "top3_predictions": top3_predictions,
                    "status": "success",
                }

                print(
                    f"字体预测结果: {font_name} (ID: {font_index}, 置信度: {font_confidence:.4f})"
                )
                return result

        except Exception as e:
            print(f"字体预测失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "font": "UNKNOWN",
                "font_confidence": 0.0,
                "font_index": -1,
                "confidence": 0.0,
                "top3_predictions": [],
                "status": "error",
                "error_message": str(e),
            }

    def predict_multiple_blocks(self, full_image, blocks):
        """批量预测多个文本块

        Args:
            full_image: 完整的原始图像（PIL Image）
            blocks: 文本块列表，每个block包含boundingBox.vertices

        Returns:
            包含所有块预测结果和统计信息的字典
        """
        detailed_results = []
        successful_blocks = 0

        for i, block in enumerate(blocks):
            try:
                # 获取边界框顶点
                vertices = block.get("boundingBox", {}).get("vertices", [])
                if len(vertices) < 4:
                    print(f"Block {i}: 顶点数量不足")
                    detailed_results.append(
                        self._create_error_result(
                            block, f"顶点数量不足: {len(vertices)}"
                        )
                    )
                    continue

                # 提取坐标
                x_coords = [v.get("x", 0) for v in vertices]
                y_coords = [v.get("y", 0) for v in vertices]

                left = max(0, min(x_coords))
                top = max(0, min(y_coords))
                right = min(full_image.width, max(x_coords))
                bottom = min(full_image.height, max(y_coords))

                width = right - left
                height = bottom - top

                # 检查边界框是否有效
                if width <= 5 or height <= 5:
                    print(f"Block {i}: 边界框太小 ({width}x{height})")
                    detailed_results.append(
                        self._create_error_result(
                            block, f"边界框太小: {width}x{height}"
                        )
                    )
                    continue

                # 添加边距（5像素）
                margin = 5
                left = max(0, left - margin)
                top = max(0, top - margin)
                right = min(full_image.width, right + margin)
                bottom = min(full_image.height, bottom + margin)

                # 裁剪图像
                crop_img = full_image.crop((left, top, right, bottom))

                # 预测字体
                font_result = self.predict_single_image(crop_img)

                # 构建响应格式
                block_info = block.copy()
                block_info["rect"] = {
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                }

                # 提取文本内容（如果存在）
                if "text" in block:
                    block_info["text"] = block["text"]
                elif "raw" in block and "words" in block["raw"]:
                    # 尝试从words中提取文本
                    text_parts = []
                    for word in block["raw"]["words"]:
                        if "symbols" in word:
                            word_text = "".join(
                                [s.get("text", "") for s in word["symbols"]]
                            )
                            text_parts.append(word_text)
                    if text_parts:
                        block_info["text"] = " ".join(text_parts)

                if font_result["status"] == "success":
                    detailed_results.append(
                        {
                            "font": font_result["font"],
                            "font_confidence": float(font_result["font_confidence"]),
                            "font_index": font_result["font_index"],
                            "confidence": float(font_result["confidence"]),
                            "block_info": block_info,
                        }
                    )
                    successful_blocks += 1
                    print(
                        f"Block {i}: 成功识别字体 '{font_result['font']}' (置信度: {font_result['font_confidence']:.4f})"
                    )
                else:
                    detailed_results.append(
                        {
                            "font": "UNKNOWN",
                            "font_confidence": 0.0,
                            "font_index": -1,
                            "confidence": 0.0,
                            "block_info": block_info,
                            "error": font_result.get("error_message", "字体识别失败"),
                        }
                    )
                    print(f"Block {i}: 字体识别失败")

            except Exception as e:
                print(f"Block {i} 处理失败: {e}")
                import traceback

                traceback.print_exc()
                detailed_results.append(self._create_error_result(block, str(e)))

        # 计算统计信息
        statistics = self._calculate_statistics(detailed_results)

        return {"detailed_results": detailed_results, "statistics": statistics}

    def _create_error_result(self, block, error_message):
        """创建错误结果

        Args:
            block: 原始block
            error_message: 错误信息

        Returns:
            错误结果字典
        """
        block_info = block.copy()
        block_info["error"] = error_message

        return {
            "font": "UNKNOWN",
            "font_confidence": 0.0,
            "font_index": -1,
            "confidence": 0.0,
            "block_info": block_info,
            "error": error_message,
        }

    def _calculate_statistics(self, detailed_results):
        """计算统计信息

        Args:
            detailed_results: 详细结果列表

        Returns:
            统计信息字典
        """
        if not detailed_results:
            return {
                "average_font_size": 0,
                "font_varieties": 0,
                "total_blocks": 0,
                "successful_blocks": 0,
                "success_rate": 0,
            }

        total_blocks = len(detailed_results)
        successful_blocks = sum(
            1 for r in detailed_results if r.get("font") != "UNKNOWN"
        )

        # 提取所有成功识别的字体
        successful_fonts = [
            r["font"] for r in detailed_results if r.get("font") != "UNKNOWN"
        ]
        font_varieties = len(set(successful_fonts))

        # 计算平均置信度
        confidences = [
            r["font_confidence"] for r in detailed_results if r.get("font") != "UNKNOWN"
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 计算字体分布
        font_distribution = {}
        for r in detailed_results:
            font = r.get("font", "UNKNOWN")
            if font != "UNKNOWN":
                font_distribution[font] = font_distribution.get(font, 0) + 1

        return {
            "average_font_size": 0,  # 这里需要根据实际需求计算字体大小
            "font_varieties": font_varieties,
            "total_blocks": total_blocks,
            "successful_blocks": successful_blocks,
            "success_rate": successful_blocks / total_blocks if total_blocks > 0 else 0,
            "average_confidence": avg_confidence,
            "font_distribution": font_distribution,
        }

    def encode_image(self, image):
        """使用SCAE编码图像（提取特征）

        Args:
            image: PIL Image对象

        Returns:
            编码后的特征向量
        """
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                # 使用SCAE编码器提取特征
                features = self.scae_model.encoder(input_tensor)
                # 展平特征
                features_flat = features.view(features.size(0), -1)

                return {
                    "features": features_flat.cpu().numpy().tolist()[0],
                    "feature_shape": list(features.shape[1:]),
                    "status": "success",
                }

        except Exception as e:
            print(f"图像编码失败: {e}")
            return {"features": [], "status": "error", "error_message": str(e)}

    def reconstruct_image(self, image):
        """使用SCAE重建图像

        Args:
            image: PIL Image对象

        Returns:
            重建后的图像（base64编码）
        """
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                # 使用SCAE进行重建
                reconstructed = self.scae_model(input_tensor)

                # 将tensor转换回PIL图像
                reconstructed = reconstructed.squeeze(0).cpu()
                reconstructed_img = tensor_to_pil(reconstructed)

                # 转换为base64
                buffered = BytesIO()
                reconstructed_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                return {
                    "reconstructed_image": f"data:image/png;base64,{img_str}",
                    "status": "success",
                }

        except Exception as e:
            print(f"图像重建失败: {e}")
            return {
                "reconstructed_image": "",
                "status": "error",
                "error_message": str(e),
            }


def resize_with_padding(pil_img, target_width, target_height, background_color=255):
    """通过添加padding调整图片大小，保持原图比例

    Args:
        pil_img: PIL图片对象
        target_width: 目标宽度
        target_height: 目标高度
        background_color: 背景颜色（灰度值）

    Returns:
        调整后的图片
    """
    # 如果原图已经大于等于目标尺寸，直接返回原图
    if pil_img.width >= target_width and pil_img.height >= target_height:
        return pil_img

    # 创建空白图片
    if pil_img.mode == "RGB":
        # 如果是RGB，转换为灰度
        pil_img = pil_img.convert("L")

    blank_img = Image.new("L", (target_width, target_height), background_color)

    # 计算粘贴位置（居中）
    x = (target_width - pil_img.width) // 2
    y = (target_height - pil_img.height) // 2

    # 粘贴图片
    blank_img.paste(pil_img, (x, y))

    return blank_img


def tensor_to_pil(tensor):
    """将tensor转换为PIL图像

    Args:
        tensor: 图像tensor

    Returns:
        PIL图像
    """
    # 移除batch维度
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)

    # 如果是单通道，添加通道维度
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)

    # 转换为numpy array
    if tensor.shape[0] == 1:  # 单通道
        img_array = tensor.squeeze(0).cpu().numpy() * 255
        img_array = img_array.astype(np.uint8)
        return Image.fromarray(img_array, mode="L")
    elif tensor.shape[0] == 3:  # 三通道
        img_array = tensor.permute(1, 2, 0).cpu().numpy() * 255
        img_array = img_array.astype(np.uint8)
        return Image.fromarray(img_array, mode="RGB")
    else:
        raise ValueError(f"不支持的tensor形状: {tensor.shape}")


# Flask应用
app = Flask(__name__)
font_service = None


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    status_info = {
        "status": "healthy" if font_service else "initializing",
        "model_type": "SCAE + CNN (Transfer Learning)",
        "scae_loaded": font_service.scae_model is not None if font_service else False,
        "cnn_loaded": font_service.cnn_model is not None if font_service else False,
        "num_fonts": font_service.num_fonts if font_service else 0,
        "device": str(font_service.device) if font_service else "unknown",
    }
    return jsonify(status_info)


@app.route("/font-recognize", methods=["POST"])
def font_recognize():
    """字体识别接口 - 兼容前端格式

    请求格式：
    {
        "image": "base64编码的图像数据",
        "blocks": [
            {
                "boundingBox": {
                    "vertices": [
                        {"x": 100, "y": 100},
                        {"x": 200, "y": 100},
                        {"x": 200, "y": 150},
                        {"x": 100, "y": 150}
                    ]
                },
                "text": "文本内容"  # 可选
            }
        ]
    }
    """
    try:
        data = request.json
        image_data = data.get("image")
        blocks = data.get("blocks", [])

        if not image_data:
            return jsonify({"error": "未提供图像数据"}), 400

        # 解码base64图像
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 批量预测多个文本块
        result = font_service.predict_multiple_blocks(image, blocks)

        return jsonify(result)

    except Exception as e:
        print(f"字体识别API错误: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict-single", methods=["POST"])
def predict_single():
    """单张图片预测接口

    请求格式：
    {
        "image": "base64编码的图像数据"
    }
    """
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "未提供图像数据"}), 400

        # 解码base64图像
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 预测字体
        result = font_service.predict_single_image(image)

        return jsonify(result)

    except Exception as e:
        print(f"单张图片预测错误: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/encode", methods=["POST"])
def encode_image():
    """图像编码接口（提取特征）

    请求格式：
    {
        "image": "base64编码的图像数据"
    }
    """
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "未提供图像数据"}), 400

        # 解码base64图像
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 编码图像
        result = font_service.encode_image(image)

        return jsonify(result)

    except Exception as e:
        print(f"图像编码API错误: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/reconstruct", methods=["POST"])
def reconstruct_image():
    """图像重建接口

    请求格式：
    {
        "image": "base64编码的图像数据"
    }
    """
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "未提供图像数据"}), 400

        # 解码base64图像
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 重建图像
        result = font_service.reconstruct_image(image)

        return jsonify(result)

    except Exception as e:
        print(f"图像重建API错误: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """批量预测接口

    请求格式：
    {
        "images": ["base64编码的图像数据1", "base64编码的图像数据2", ...]
    }
    """
    try:
        data = request.json
        images_data = data.get("images", [])

        if not images_data:
            return jsonify({"error": "未提供图像数据"}), 400

        results = []
        for i, image_data in enumerate(images_data):
            try:
                # 解码base64图像
                if image_data.startswith("data:"):
                    image_data = image_data.split(",")[1]

                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))

                # 预测字体
                result = font_service.predict_single_image(image)
                result["image_index"] = i
                results.append(result)

            except Exception as e:
                results.append(
                    {
                        "image_index": i,
                        "font": "UNKNOWN",
                        "font_confidence": 0.0,
                        "font_index": -1,
                        "confidence": 0.0,
                        "top3_predictions": [],
                        "status": "error",
                        "error_message": str(e),
                    }
                )

        return jsonify(
            {
                "results": results,
                "total_images": len(images_data),
                "success_count": sum(
                    1 for r in results if r.get("status") == "success"
                ),
            }
        )

    except Exception as e:
        print(f"批量预测API错误: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """获取模型信息接口"""
    if not font_service:
        return jsonify({"error": "服务未初始化"}), 500

    # 统计模型参数
    scae_params = sum(p.numel() for p in font_service.scae_model.parameters())
    cnn_params = sum(p.numel() for p in font_service.cnn_model.parameters())

    return jsonify(
        {
            "scae_parameters": scae_params,
            "cnn_parameters": cnn_params,
            "total_parameters": scae_params + cnn_params,
            "num_fonts": font_service.num_fonts,
            "device": str(font_service.device),
            "fonts_list": list(font_service.font_dict.keys())[:10],  # 只显示前10种字体
        }
    )


@app.route("/compatibility-test", methods=["POST"])
def compatibility_test():
    """兼容性测试接口，用于测试前端响应格式"""
    try:
        data = request.json
        image_data = data.get("image")
        blocks = data.get("blocks", [])

        if not image_data:
            return jsonify({"error": "未提供图像数据"}), 400

        # 创建模拟响应
        detailed_results = []
        for i, block in enumerate(blocks[:3]):  # 只处理前3个块作为示例
            detailed_results.append(
                {
                    "font": f"Font_{i % 5}",
                    "font_confidence": 0.85 + (i * 0.05),
                    "font_index": i % 5,
                    "confidence": 0.85 + (i * 0.05),
                    "block_info": block,
                }
            )

        statistics = {
            "average_font_size": 16.5,
            "font_varieties": 3,
            "total_blocks": len(blocks),
            "successful_blocks": min(3, len(blocks)),
            "success_rate": min(3, len(blocks)) / len(blocks) if len(blocks) > 0 else 0,
        }

        return jsonify({"detailed_results": detailed_results, "statistics": statistics})

    except Exception as e:
        print(f"兼容性测试错误: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        # 模型路径配置 - 根据你的实际情况修改
        scae_model_path = "./dataset/models/SCAE_en_352.pth"  # SCAE模型
        cnn_model_path = "./dataset/models/CNN_en_352.pth"  # CNN模型
        font_dict_path = "./dataset/models/font_dict_en_352.json"  # 字体字典

        print("正在初始化字体识别服务...")
        print(f"SCAE模型路径: {scae_model_path}")
        print(f"CNN模型路径: {cnn_model_path}")
        print(f"字体字典路径: {font_dict_path}")

        # 初始化服务
        font_service = FontRecognitionService(
            scae_model_path=scae_model_path,
            cnn_model_path=cnn_model_path,
            font_dict_path=font_dict_path,
            device=None,  # 自动选择设备
        )

        print("字体识别服务启动成功")
        print("模型架构: SCAE + CNN (迁移学习)")
        print(f"支持字体数量: {font_service.num_fonts}")
        print("可用API接口:")
        print("  GET  /health             - 健康检查")
        print("  POST /font-recognize     - 多文本块字体识别（兼容前端）")
        print("  POST /predict-single     - 单张图片字体识别")
        print("  POST /encode             - 图像特征提取")
        print("  POST /reconstruct        - 图像重建")
        print("  POST /batch-predict      - 批量预测")
        print("  GET  /model-info         - 模型信息")
        print("  POST /compatibility-test - 兼容性测试")

        # 启动Flask应用
        app.run(host="0.0.0.0", port=5000, debug=False)

    except Exception as e:
        print(f"服务启动失败: {e}")
        import traceback

        traceback.print_exc()
