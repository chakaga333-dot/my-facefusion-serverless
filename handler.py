import runpod
import subprocess
import os
import base64
import urllib.request

def download_input(url, output_path):
    """Декодирует Base64 от пользователя или скачивает URL"""
    if url.startswith("data:image") or ";base64," in url:
        print("📦 Обработка фото из интерфейса (Base64)...")
        base64_data = url.split(",")[1] if "," in url else url
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        return output_path
    
    print(f"📥 Скачивание файла: {url}")
    urllib.request.urlretrieve(url, output_path)
    return output_path

def handler(job):
    try:
        job_input = job["input"]
        source_url = job_input.get("source")
        # ТВОЁ ВИДЕО: Вставь сюда прямую ссылку на свой шаблон
        target_url = job_input.get("target", "ССЫЛКА_НА_ТВОЕ_ВИДЕО")
        
        os.makedirs("/tmp/input", exist_ok=True)
        os.makedirs("/tmp/output", exist_ok=True)
        
        source_path = "/tmp/input/source.jpg"
        target_path = "/tmp/input/target.mp4"
        output_path = "/tmp/output/result.mp4"
        
        # Загружаем файлы
        download_input(source_url, source_path)
        if not os.path.exists(target_path): # Скачиваем шаблон только если его нет
            urllib.request.urlretrieve(target_url, target_path)
        
        # Команда запуска (используем нашу "лучшую сборку")
        command = [
            "python", "facefusion.py", "headless-run",
            "-s", source_path, "-t", target_path, "-o", output_path,
            "--processors", "face_swapper",
            "--execution-providers", "cuda"
        ]
        
        print("🚀 Запуск генерации на RTX 4090...")
        subprocess.run(command, cwd="/app", check=True)
        
        # КОНВЕРТАЦИЯ РЕЗУЛЬТАТА В BASE64
        with open(output_path, "rb") as video_file:
            encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
            
        return {
            "success": True,
            "video_base64": encoded_video,
            "message": "круто" # Твое ключевое слово
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})