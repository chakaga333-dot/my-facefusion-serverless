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
else:
    print("✅ NumPy версия корректная")

providers = onnxruntime.get_available_providers()
print(f"Доступные провайдеры: {providers}")
print("CUDA доступна:", "CUDAExecutionProvider" in providers)
print("=" * 60)
sys.stdout.flush()

def save_file_from_url(url, output_path):
    try:
        print(f"📥 Скачиваю файл: {url} -> {output_path}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        return False

def send_callback(url, data):
    try:
        requests.post(url, json=data, timeout=30)
        print(f"📡 Callback отправлен на {url}")
    except Exception as e:
        print(f"⚠️ Ошибка callback: {e}")

def handler(job):
    try:
        job_input = job["input"]
        request_id = job.get("id")
        
        # Параметры для напарника
        user_id = job_input.get("userId")
        callback_url = job_input.get("callbackUrl")
        
        # 1. ПОДГОТОВКА ПУТЕЙ
        source_path = "/tmp/source.jpg"
        target_path = "/tmp/target.mp4"
        output_path = "/tmp/output_result.mp4"

        # 2. ПОЛУЧЕНИЕ ИСТОЧНИКА (ЛИЦО)
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))
            print("✅ Лицо получено из Base64")
        else:
            return {"success": False, "error": "❌ Нет faceBase64"}

        # 3. ПОЛУЧЕНИЕ ТАРГЕТА (ВИДЕО)
        template_url = job_input.get("templateUrl")
        if template_url:
            if template_url.startswith("/workspace"):
                target_path = template_url # Используем локальный путь
                print(f"📂 Использую локальное видео: {target_path}")
            else:
                if not save_file_from_url(template_url, target_path):
                    return {"success": False, "error": "❌ Не удалось скачать видео"}
        else:
            return {"success": False, "error": "❌ Не указано видео (templateUrl)"}

        # 4. ФОРМИРОВАНИЕ КОМАНДЫ (ИСПРАВЛЕННО)
        # Мы принудительно добавляем --content-analyser-model none
        args = [
            "python", "facefusion.py", "headless-run",
            "-s", source_path,
            "-t", target_path,
            "-o", output_path,
            "--processors", "face_swapper",
            "--execution-providers", "cuda",
            "--video-memory-strategy", "moderate",
            "--face-detector-model", "yoloface",
            "--skip-download",
            "--content-analyser-model", "none" # ЭТО ГЛАВНЫЙ ФИКС
        ]

        # Если в запросе пришли свои доп. аргументы, объединяем их осторожно
        extra_args = job_input.get("args", [])
        if extra_args and isinstance(extra_args, list):
            # Убираем из входящих args те, что могут конфликтовать
            cleaned_extra = [a for a in extra_args if a not in args and a != "facefusion.py" and a != "headless-run"]
            args.extend(cleaned_extra)

        print(f"🚀 ЗАПУСК FACEFUSION: {' '.join(args)}")
        sys.stdout.flush()

        # 5. ВЫПОЛНЕНИЕ
        result = subprocess.run(args, cwd="/app", capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_log = result.stderr or result.stdout
            print(f"❌ ОШИБКА FACEFUSION:\n{error_log}")
            return {"success": False, "error": error_log}

        # 6. КОДИРОВАНИЕ РЕЗУЛЬТАТА
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_base64 = base64.b64encode(v.read()).decode('utf-8')
            
            response_data = {
                "success": True,
                "videoBase64": video_base64,
                "requestId": request_id,
                "message": "круто"
            }
            
            if callback_url:
                send_callback(callback_url, response_data)
            
            return response_data
        else:
            return {"success": False, "error": "❌ Файл результата не найден"}

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        # Чистка временных файлов
        for p in [source_path, output_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})