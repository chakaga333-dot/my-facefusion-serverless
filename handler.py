import runpod
import subprocess
import os
import base64
import urllib.request
import requests

def handler(job):
    try:
        job_input = job["input"]
        requestId = job_input.get("requestId", "task_1")
        
        # Пути (используем стандартные из твоего рабочего примера)
        source_path = "/tmp/source.jpg"
        target_path = job_input.get("templatePath", "/tmp/target.mp4")
        output_path = "/tmp/output_result.mp4"

        # 1. Сохраняем фото (Base64 от HTML или напарника)
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))

        # 2. Проверяем шаблон (если напарник прислал URL)
        template_url = job_input.get("templateUrl")
        if template_url and not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            urllib.request.urlretrieve(template_url, target_path)

        # 3. ТВОЯ ИДЕАЛЬНАЯ КОМАНДА
        command = [
            "python", "facefusion.py",
            "headless-run",
            "-s", source_path,
            "-t", target_path,
            "-o", output_path,
            "--processors", "face_swapper",
            "--execution-providers", "cuda",
            "--execution-thread-count", "4",
            "--execution-queue-count", "2",
            "--video-memory-strategy", "moderate",
            "--face-detector-model", "yoloface",
            "--face-detector-size", "640x640",
            "--skip-download" # Добавляем, чтобы не качал модели из сети
        ]

        print(f"🚀 Running command: {' '.join(command)}")
        
        # Запуск
        result = subprocess.run(command, cwd="/app", capture_output=True, text=True)

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        # 4. Кодируем результат в Base64 для возврата в HTML/напарнику
        video_data = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')

        return {
            "success": True,
            "videoBase64": video_data,
            "message": "круто"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})