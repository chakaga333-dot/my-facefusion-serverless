import runpod
import subprocess
import os
import base64
import sys

def handler(job):
    try:
        job_input = job["input"]
        
        # 1. ОТКЛЮЧАЕМ NSFW ЧЕРЕЗ ПЕРЕМЕННУЮ ОКРУЖЕНИЯ
        # Это не даст ему проверять хэш модели open_nsfw
        os.environ["FACEFUSION_CONTENT_ANALYSER_MODEL"] = "none"
        
        # Также создаем конфиг на всякий случай
        config_dir = os.path.expanduser('~/.facefusion')
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, 'facefusion.ini'), 'w') as f:
            f.write('[choices]\ncontent_analyser_model = none\n')

        # 2. ПУТИ
        source_path = "/tmp/source.jpg"
        target_path = job_input.get("targetPath", "/workspace/video/1.mp4")
        output_path = "/tmp/output_result.mp4"

        # 3. СОХРАНЯЕМ ЛИЦО (Base64 -> Файл)
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))
        else:
            return {"success": False, "error": "No faceBase64 provided"}

        # 4. ТВОЯ КОМАНДА (БЕЗ ОШИБОЧНОГО АРГУМЕНТА)
        # Убрали --content-analyser-model чтобы facefusion не ругался
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
            "--skip-download"
        ]

        print(f"🚀 GPU Task Start: {' '.join(command)}")
        sys.stdout.flush()
        
        # Запуск FaceFusion
        result = subprocess.run(command, cwd="/app", capture_output=True, text=True)

        # 5. ПРОВЕРЯЕМ РЕЗУЛЬТАТ И ОТПРАВЛЯЕМ BASE64
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')
            
            # Удаляем временные файлы
            os.remove(source_path)
            os.remove(output_path)

            return {
                "success": True,
                "videoBase64": video_data, # Твое видео летит в HTML!
                "message": "круто"
            }
        else:
            # Если файла нет, возвращаем логи ошибки
            return {
                "success": False, 
                "error": "Файл не создался. Проверь логи.",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})