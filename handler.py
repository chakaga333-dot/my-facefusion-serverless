import runpod
import subprocess
import os
import sys
import base64
import urllib.request
import requests
import onnxruntime

# ============================================================
# ДИАГНОСТИКА CUDA ПРИ ЗАПУСКЕ
# ============================================================
print("=" * 60)
print("🔍 ДИАГНОСТИКА ONNX RUNTIME")
print("=" * 60)

import numpy as np
print(f"NumPy версия: {np.__version__}")
if np.__version__.startswith('2.'):
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: NumPy 2.x установлена!")
    print("   Требуется NumPy 1.26.4")
else:
    print("✅ NumPy версия корректная")

providers = onnxruntime.get_available_providers()
print(f"ONNX Runtime версия: {onnxruntime.__version__}")
print("Доступные провайдеры:", providers)
print("CUDA доступна:", "CUDAExecutionProvider" in providers)
print("=" * 60)

print("📋 ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '❌ Не установлена')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '❌ Не установлена')}")
print("=" * 60)
sys.stdout.flush()


def save_file_from_url(url, output_path):
    """Скачивание файла по URL"""
    try:
        print(f"📥 Скачиваю файл: {url}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Файл сохранен: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {str(e)}")
        return False


def save_file_from_base64(base64_data, output_path):
    """Сохранение base64 в файл"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(base64_data))
        print(f"✅ Файл сохранен из base64: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения base64: {str(e)}")
        return False


def file_to_base64(file_path):
    """Конвертация файла в base64"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        base64_data = base64.b64encode(data).decode('utf-8')
        print(f"✅ Файл конвертирован в base64 ({len(base64_data)} символов)")
        return base64_data
    except Exception as e:
        print(f"❌ Ошибка конвертации: {str(e)}")
        return None


def send_callback(callback_url, data):
    """Отправка callback на VPS сервер"""
    try:
        print(f"📡 Отправка callback на {callback_url}")
        response = requests.post(callback_url, json=data, timeout=30)
        print(f"✅ Callback отправлен: {response.status_code}")
        return True
    except Exception as e:
        print(f"⚠️ Ошибка callback: {str(e)}")
        return False


def handler(job):
    """
    Универсальный обработчик с поддержкой GPU и callback
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 НАЧАЛО ОБРАБОТКИ ЗАДАЧИ")
        print("=" * 60)
        
        job_input = job["input"]
        
        request_id = job_input.get("requestId", "unknown")
        user_id = job_input.get("userId", "unknown")
        callback_url = job_input.get("callbackUrl")
        
        os.makedirs("/tmp/input", exist_ok=True)
        os.makedirs("/tmp/output", exist_ok=True)
        
        # 1. Template/Target Video
        template_url = job_input.get("templateUrl")
        template_path = job_input.get("templatePath", "/tmp/input/target.mp4")
        target_url = job_input.get("target_video_url")
        
        video_source = template_url or target_url
        if not video_source:
            return {"success": False, "error": "❌ Не указано видео"}
        
        if not video_source.startswith("/workspace"): # Если не локальный путь, качаем
            if not save_file_from_url(video_source, template_path):
                return {"success": False, "error": "❌ Не удалось скачать видео"}
        else:
            template_path = video_source

        # 2. Face Image (Source)
        face_url = job_input.get("faceUrl")
        face_base64 = job_input.get("faceBase64")
        source_image_b64 = job_input.get("source_image")
        face_path = job_input.get("facePath", "/tmp/input/source.jpg")
        
        if face_url:
            save_file_from_url(face_url, face_path)
        elif face_base64 or source_image_b64:
            save_file_from_base64(face_base64 or source_image_b64, face_path)
        else:
            return {"success": False, "error": "❌ Нет лица"}
        
        output_path = job_input.get("outputPath", "/tmp/output/result.mp4")
        
        # ==================================================
        # ИЗМЕНЕННЫЙ БЛОК: ФОРМИРОВАНИЕ КОМАНДЫ
        # ==================================================
        custom_args = job_input.get("args")
        
        if custom_args:
            # Если напарник прислал свои аргументы, мы принудительно добавляем отключение NSFW в конец
            if "--content-analyser-model" not in custom_args:
                custom_args.extend(["--content-analyser-model", "none"])
            command = ["python"] + custom_args
        else:
            # Твоя идеальная GPU команда
            command = [
                "python", "facefusion.py",
                "headless-run",
                "-s", face_path,
                "-t", template_path,
                "-o", output_path,
                "--processors", "face_swapper",
                "--execution-providers", "cuda", 
                "--execution-thread-count", "4",
                "--execution-queue-count", "2",
                "--video-memory-strategy", "moderate",
                "--face-detector-model", "yoloface",
                "--face-detector-size", "640x640",
                "--skip-download",
                # ГЛАВНОЕ ДОПОЛНЕНИЕ:
                "--content-analyser-model", "none" 
            ]
        
        print("\n🔧 КОМАНДА ЗАПУСКА:")
        print(" ".join(command))
        sys.stdout.flush()
        
        result = subprocess.run(command, cwd="/app", capture_output=True, text=True, timeout=600)
        
        # ... (весь остальной код возврата результата остается БЕЗ изменений) ...
        print("\n📄 STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            error_data = {"requestId": request_id, "success": False, "error": result.stderr}
            if callback_url: send_callback(callback_url, error_data)
            return error_data
        
        if not os.path.exists(output_path):
            return {"success": False, "error": "❌ Файл не создан"}
        
        video_base64 = file_to_base64(output_path)
        response_data = {
            "requestId": request_id,
            "userId": user_id,
            "success": True,
            "videoBase64": video_base64,
            "file_size_mb": round(os.path.getsize(output_path) / 1024 / 1024, 2),
            "message": "Обработка успешно завершена"
        }
        
        if callback_url:
            send_callback(callback_url, response_data)
            response_data["videoBase64"] = None
        
        return response_data
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})