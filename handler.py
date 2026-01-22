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
    
    Поддерживает два режима работы:
    1. Простой режим (простые параметры)
    2. Расширенный режим напарника (с args, callback, templateUrl и т.д.)
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 НАЧАЛО ОБРАБОТКИ ЗАДАЧИ")
        print("=" * 60)
        
        job_input = job["input"]
        
        # Извлечение параметров
        request_id = job_input.get("requestId", "unknown")
        user_id = job_input.get("userId", "unknown")
        callback_url = job_input.get("callbackUrl")
        
        print(f"🎬 Request ID: {request_id}")
        print(f"👤 User ID: {user_id}")
        if callback_url:
            print(f"📞 Callback URL: {callback_url}")
        
        # Создание временных директорий
        os.makedirs("/tmp/input", exist_ok=True)
        os.makedirs("/tmp/output", exist_ok=True)
        
        # ==================================================
        # ОБРАБОТКА ВХОДНЫХ ФАЙЛОВ
        # ==================================================
        
        # 1. Template/Target Video
        template_url = job_input.get("templateUrl")
        template_path = job_input.get("templatePath", "/tmp/input/target.mp4")
        target_url = job_input.get("target_video_url")  # Альтернативное название
        
        video_source = template_url or target_url
        if not video_source:
            error_msg = "❌ Не указано видео (templateUrl или target_video_url)"
            print(error_msg)
            if callback_url:
                send_callback(callback_url, {
                    "requestId": request_id,
                    "userId": user_id,
                    "success": False,
                    "error": error_msg
                })
            return {"success": False, "error": error_msg}
        
        if not save_file_from_url(video_source, template_path):
            error_msg = "❌ Не удалось скачать видео"
            if callback_url:
                send_callback(callback_url, {
                    "requestId": request_id,
                    "userId": user_id,
                    "success": False,
                    "error": error_msg
                })
            return {"success": False, "error": error_msg}
        
        # 2. Face Image (Source)
        face_url = job_input.get("faceUrl")
        face_base64 = job_input.get("faceBase64")
        source_image_b64 = job_input.get("source_image")  # Альтернативное название
        
        face_path = job_input.get("facePath", "/tmp/input/source.jpg")
        
        # Приоритет: faceUrl > faceBase64 > source_image
        if face_url:
            if not save_file_from_url(face_url, face_path):
                error_msg = "❌ Не удалось скачать изображение лица"
                if callback_url:
                    send_callback(callback_url, {
                        "requestId": request_id,
                        "userId": user_id,
                        "success": False,
                        "error": error_msg
                    })
                return {"success": False, "error": error_msg}
        elif face_base64 or source_image_b64:
            base64_data = face_base64 or source_image_b64
            if not save_file_from_base64(base64_data, face_path):
                error_msg = "❌ Не удалось сохранить изображение из base64"
                if callback_url:
                    send_callback(callback_url, {
                        "requestId": request_id,
                        "userId": user_id,
                        "success": False,
                        "error": error_msg
                    })
                return {"success": False, "error": error_msg}
        else:
            error_msg = "❌ Не указано изображение лица"
            if callback_url:
                send_callback(callback_url, {
                    "requestId": request_id,
                    "userId": user_id,
                    "success": False,
                    "error": error_msg
                })
            return {"success": False, "error": error_msg}
        
        # ==================================================
        # ЗАПУСК FACEFUSION
        # ==================================================
        
        output_path = job_input.get("outputPath", "/tmp/output/result.mp4")
        
        # Проверяем, передал ли напарник custom args
        custom_args = job_input.get("args")
        
        if custom_args:
            # Используем args от напарника
            print(f"🔧 Используются custom args от сервера")
            command = ["python"] + custom_args
        else:
            # Используем нашу оптимизированную GPU команду
            # ВАЖНО: Отключаем content analyser чтобы избежать ошибки open_nsfw
            command = [
                "python", "facefusion.py",
                "headless-run",
                "-s", face_path,
                "-t", template_path,
                "-o", output_path,
                "--processors", "face_swapper",  # ТОЛЬКО face_swapper, БЕЗ content_analyser
                "--execution-providers", "cuda",
                "--execution-thread-count", "4",
                "--execution-queue-count", "2",
                "--video-memory-strategy", "moderate",
                "--face-detector-model", "yoloface",
                "--face-detector-size", "640x640",
                "--output-video-encoder", "libx264",  # Стандартный кодек
                "--output-video-quality", "80",  # Хорошее качество
                "--skip-audio"  # Пропускаем аудио для ускорения
            ]
        
        print("\n🔧 КОМАНДА ЗАПУСКА:")
        print(" ".join(command))
        print("\n⏳ Обработка началась (макс. 10 минут)...")
        sys.stdout.flush()
        
        # Запуск FaceFusion
        result = subprocess.run(
            command,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # Вывод логов
        print("\n📄 STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)
        
        sys.stdout.flush()
        
        # Проверка результата
        if result.returncode != 0:
            error_data = {
                "requestId": request_id,
                "userId": user_id,
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
            
            if callback_url:
                send_callback(callback_url, error_data)
            
            return error_data
        
        # ==================================================
        # ОБРАБОТКА РЕЗУЛЬТАТА
        # ==================================================
        
        if not os.path.exists(output_path):
            error_msg = "❌ Выходной файл не был создан"
            if callback_url:
                send_callback(callback_url, {
                    "requestId": request_id,
                    "userId": user_id,
                    "success": False,
                    "error": error_msg
                })
            return {"success": False, "error": error_msg}
        
        file_size = os.path.getsize(output_path)
        print(f"\n✅ УСПЕХ! Файл создан: {output_path}")
        print(f"📦 Размер файла: {file_size / 1024 / 1024:.2f} MB")
        
        # Конвертация в base64
        print("\n🔄 Конвертация видео в base64...")
        video_base64 = file_to_base64(output_path)
        
        if not video_base64:
            error_msg = "❌ Не удалось конвертировать видео в base64"
            if callback_url:
                send_callback(callback_url, {
                    "requestId": request_id,
                    "userId": user_id,
                    "success": False,
                    "error": error_msg
                })
            return {"success": False, "error": error_msg}
        
        # Подготовка ответа
        response_data = {
            "requestId": request_id,
            "userId": user_id,
            "success": True,
            "videoBase64": video_base64,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "message": "Обработка успешно завершена"
        }
        
        # Отправка callback если указан URL
        if callback_url:
            send_callback(callback_url, response_data)
            # Для callback режима не возвращаем base64 в основном ответе (экономия)
            response_data["videoBase64"] = None
            response_data["message"] += " (отправлено через callback)"
        
        # Очистка временных файлов
        try:
            if os.path.exists(face_path):
                os.remove(face_path)
            if os.path.exists(template_path):
                os.remove(template_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            print("🧹 Временные файлы удалены")
        except Exception as e:
            print(f"⚠️ Ошибка очистки: {e}")
        
        return response_data
        
    except subprocess.TimeoutExpired:
        error_msg = "⏱️ Превышен таймаут обработки (10 минут)"
        print(error_msg)
        
        if callback_url:
            send_callback(callback_url, {
                "requestId": request_id,
                "userId": user_id,
                "success": False,
                "error": error_msg
            })
        
        return {"success": False, "error": error_msg}
        
    except Exception as e:
        error_msg = f"❌ Неожиданная ошибка: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        if callback_url:
            send_callback(callback_url, {
                "requestId": request_id,
                "userId": user_id,
                "success": False,
                "error": error_msg
            })
        
        return {"success": False, "error": error_msg}


# ============================================================
# ЗАПУСК RUNPOD SERVERLESS HANDLER
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 ЗАПУСК UNIFIED FACEFUSION HANDLER (GPU + CALLBACK)")
    print("=" * 60)
    sys.stdout.flush()
    
    runpod.serverless.start({"handler": handler})