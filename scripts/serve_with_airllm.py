"""
AirLLM 推理服务

使用 AirLLM 部署简易 API 推理服务。
由于 AirLLM 的分层加载特性，可以在低显存环境下提供服务。

使用:
    python scripts/serve_with_airllm.py --model_dir models/lmm_small_hf --port 8000

API 接口:
    POST /generate
    {
        "prompt": "勾股定理的内容是",
        "max_new_tokens": 50,
        "temperature": 0.8
    }
"""

import argparse
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class LLMHandler(BaseHTTPRequestHandler):
    """简易 HTTP 请求处理器"""

    model = None
    tokenizer = None

    def do_POST(self):
        """处理 POST 请求"""
        if self.path == "/generate":
            self._handle_generate()
        else:
            self._send_error(404, "Not Found")

    def do_GET(self):
        """处理 GET 请求"""
        if self.path == "/health":
            self._send_json({"status": "ok", "model_loaded": self.model is not None})
        else:
            self._send_json({
                "message": "LMM AirLLM 推理服务",
                "endpoints": {
                    "POST /generate": "文本生成",
                    "GET /health": "健康检查",
                }
            })

    def _handle_generate(self):
        """处理生成请求"""
        try:
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            prompt = data.get("prompt", "")
            max_new_tokens = data.get("max_new_tokens", 50)
            temperature = data.get("temperature", 0.8)

            if not prompt:
                self._send_error(400, "Missing 'prompt' field")
                return

            # 生成文本
            if self.model is None:
                self._send_error(503, "Model not loaded")
                return

            # 使用 AirLLM 生成
            inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            input_ids = inputs['input_ids']

            generation_output = self.model.generate(
                input_ids.cuda() if hasattr(input_ids, 'cuda') else input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )

            output_text = self.tokenizer.decode(generation_output.sequences[0])

            self._send_json({
                "prompt": prompt,
                "output": output_text,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            })

        except Exception as e:
            self._send_error(500, str(e))

    def _send_json(self, data: dict):
        """发送 JSON 响应"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _send_error(self, code: int, message: str):
        """发送错误响应"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))

    def log_message(self, format, *args):
        """自定义日志（简化输出）"""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def load_model_with_airllm(model_dir: str):
    """
    使用 AirLLM 加载模型

    Args:
        model_dir: HuggingFace 格式模型目录

    Returns:
        (model, tokenizer)
    """
    print(f"使用 AirLLM 加载模型: {model_dir}")

    try:
        from airllm import AutoModel
    except ImportError:
        print("错误: 未安装 AirLLM。请运行: pip install airllm")
        return None, None

    try:
        model = AutoModel.from_pretrained(model_dir)
        tokenizer = model.tokenizer
        print("模型加载成功!")
        return model, tokenizer
    except Exception as e:
        print(f"AirLLM 加载失败: {e}")
        print("尝试使用标准 Transformers 加载...")
        return load_model_with_transformers(model_dir)


def load_model_with_transformers(model_dir: str):
    """
    使用标准 Transformers 加载模型（备用方案）

    Args:
        model_dir: HuggingFace 格式模型目录

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        print(f"模型已加载到: {device}")
        return model, tokenizer
    except Exception as e:
        print(f"Transformers 加载也失败: {e}")
        return None, None


def serve(model_dir: str, host: str = "0.0.0.0", port: int = 8000):
    """
    启动推理服务

    Args:
        model_dir: HuggingFace 格式模型目录
        host: 监听地址
        port: 监听端口
    """
    # 加载模型
    model, tokenizer = load_model_with_airllm(model_dir)

    if model is None:
        print("模型加载失败，服务无法启动")
        return

    LLMHandler.model = model
    LLMHandler.tokenizer = tokenizer

    # 启动 HTTP 服务
    server = HTTPServer((host, port), LLMHandler)
    print(f"\n服务已启动: http://{host}:{port}")
    print("API 接口:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  POST http://{host}:{port}/generate")
    print("\n示例请求:")
    print(f'  curl -X POST http://{host}:{port}/generate \\")
    print(f'    -H "Content-Type: application/json" \\")
    print(f'    -d \'{{"prompt": "勾股定理的内容是", "max_new_tokens": 50}}\'')
    print("\n按 Ctrl+C 停止服务")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="AirLLM 推理服务")
    parser.add_argument("--model_dir", type=str, required=True, help="HuggingFace 格式模型目录")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        return

    serve(args.model_dir, args.host, args.port)


if __name__ == "__main__":
    main()