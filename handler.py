import runpod
import subprocess
import os
import base64
import time

def handler(job):
    try:
        job_input = job["input"]
        
        # 1. ПРИНУДИТЕЛЬНОЕ ОТКЛЮЧЕНИЕ NSFW (через переменные окружения и конфиг)
        os.environ["FACEFUSION_CONTENT_ANALYSER_MODEL"] = "none"
        config_dir = os.path.expanduser('~/.facefusion')
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, 'facefusion.ini'), 'w') as f:
            f.write('[choices]\ncontent_analyser_model = none\n')

        # 2. ПУТИ
        source_path = "/tmp/source.jpg"
        target_path = job_input.get("targetPath", "/workspace/video/1.mp4")
        output_path = "/tmp/output_result.mp4"

        # Очистка старых файлов перед стартом
        if os.path.exists(output_path): os.remove(output_path)

        # 3. ДЕКОДИРОВАНИЕ ЛИЦА
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))
        else:
            return {"success": False, "error": "No faceBase64"}

        # 4. ТВОЯ КОМАНДА (БЕЗ СПОРНЫХ ФЛАГОВ)
        # Оставляем только то, что 100% работает в консоли
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

        print(f"🚀 СТАРТ ГЕНЕРАЦИИ: {' '.join(command)}")
        
        # Запуск с захватом всех логов для отладки
        process = subprocess.run(command, cwd="/app", capture_output=True, text=True)

        # Проверяем результат
        if not os.path.exists(output_path):
            return {
                "success": False, 
                "error": "Видео не создано", 
                "stdout": process.stdout, 
                "stderr": process.stderr
            }

        # 5. КОДИРУЕМ ВИДЕО ОБРАТНО В BASE64
        with open(output_path, "rb") as v:
            video_data = base64.b64encode(v.read()).decode('utf-8')

        # Чистим временные файлы
        os.remove(source_path)
        os.remove(output_path)

        return {
            "success": True,
            "videoBase64": video_data,
            "message": "круто"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})