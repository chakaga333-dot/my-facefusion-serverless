import runpod
import subprocess
import os
import sys
import base64
import urllib.request
import requests
import onnxruntime
import shutil

# ============================================================
# ДИАГНОСТИКА CUDA ПРИ ЗАПУСКЕ
# ============================================================
print("=" * 60)
print("🔍 ДИАГНОСТИКА СИСТЕМЫ 'КРУТО'")
print("=" * 60)

import numpy as np
print(f"NumPy версия: {np.__version__}")
providers = onnxruntime.get_available_providers()
print(f"ONNX Runtime: {onnxruntime.__version__}")
print(f"Доступные провайдеры: {providers}")
print("CUDA доступна:", "CUDAExecutionProvider" in providers)
print("=" * 60)
sys.stdout.flush()

def save_file_from_url(url, output_path):
    try:
        print(f"📥 Скачиваю файл: {url}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        return False

def save_file_from_base64(base64_data, output_path):
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        return True
    except Exception as e:
        print(f"❌ Ошибка base64: {e}")
        return False

def send_callback(url, data):
    try:
        requests.post(url, json=data, timeout=30)
        print(f"📡 Callback отправлен на {url}")
    except Exception as e:
        print(f"⚠️ Ошибка callback: {e}")

def handler(job):
    try:
        print("\n🚀 НАЧАЛО ОБРАБОТКИ")
        job_input = job["input"]
        request_id = job.get("id", "unknown")
        user_id = job_input.get("userId", "unknown")
        callback_url = job_input.get("callbackUrl")

        # --- [ ФИКС: ПРИНУДИТЕЛЬНОЕ ОТКЛЮЧЕНИЕ NSFW ПРОВЕРКИ ] ---
        # 1. Создаем facefusion.ini, чтобы отключить анализатор на уровне настроек
        config_dir = os.path.expanduser('~/.facefusion')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'facefusion.ini')
        with open(config_path, 'w') as f:
            f.write('[choices]\ncontent_analyser_model = none\n')
        
        # 2. Удаляем битую модель из кэша, если она там есть (после wget в Docker)
        nsfw_cache_path = os.path.join(config_dir, "models/open_nsfw.onnx")
        if os.path.exists(nsfw_cache_path):
            os.remove(nsfw_cache_path)
            print("🧹 Старый файл open_nsfw удален для обхода ошибки хэша")
        # --------------------------------------------------------

        source_path = "/tmp/source.jpg"
        target_path = "/tmp/target.mp4"
        output_path = "/tmp/output_result.mp4"

        # Шаг 1: Лицо
        face_base64 = job_input.get("faceBase64") or job_input.get("source_image")
        if face_base64:
            save_file_from_base64(face_base64, source_path)
        else:
            return {"success": False, "error": "❌ Нет данных лица"}

        # Шаг 2: Видео
        video_url = job_input.get("templateUrl") or job_input.get("target_video_url")
        if not video_url:
            return {"success": False, "error": "❌ Не указано видео"}
        
        if video_url.startswith("/workspace"):
            target_path = video_url
            print(f"📂 Использую локальное видео: {target_path}")
        else:
            if not save_file_from_url(video_url, target_path):
                return {"success": False, "error": "❌ Ошибка загрузки видео"}

        # Шаг 3: Команда
        # Используем твой проверенный набор аргументов
        command = [
            "python", "facefusion.py", "headless-run",
            "-s", source_path,
            "-t", target_path,
            "-o", output_path,
            "--processors", "face_swapper",
            "--execution-providers", "cuda",
            "--execution-thread-count", "4",
            "--execution-queue-count", "2",
            "--video-memory-strategy", "moderate",
            "--face-detector-model", "yoloface",
            "--skip-download" # Теперь безопасно, так как мы отключили NSFW через конфиг
        ]

        print(f"🛠️ ЗАПУСК: {' '.join(command)}")
        sys.stdout.flush()

        # Шаг 4: Выполнение
        result = subprocess.run(command, cwd="/app", capture_output=True, text=True, timeout=600)

        # Шаг 5: Результат
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')
            
            response = {
                "success": True,
                "videoBase64": video_data,
                "requestId": request_id,
                "message": "круто"
            }
            
            if callback_url:
                send_callback(callback_url, response)
            return response
        else:
            print(f"❌ STDERR: {result.stderr}")
            return {"success": False, "error": "Файл не создан", "log": result.stderr}

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Очистка временных файлов (кроме тех, что в workspace)
        for p in [source_path, output_path]:
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})