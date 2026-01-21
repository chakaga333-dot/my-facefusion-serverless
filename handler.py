import runpod
import subprocess
import os
import base64
import urllib.request

def handler(job):
    try:
        job_input = job["input"]
        
        # 1. Пути внутри твоего контейнера
        source_path = "/tmp/source.jpg"
        # Напарник сам скажет, какой шаблон взять из твоего хранилища
        target_path = job_input.get("templatePath", "/runpod-volume/templates/4.mp4")
        output_path = "/tmp/result.mp4"

        # 2. Получаем лицо (Base64 от напарника)
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))
        else:
            return {"success": False, "error": "No face data"}

        # 3. ТВОЯ ИДЕАЛЬНАЯ КОМАНДА (Чистая мощь GPU)
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
            "--skip-download"
        ]

        # Запуск FaceFusion
        print(f"🚀 GPU Task Start...")
        result = subprocess.run(command, cwd="/app", capture_output=True, text=True)

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        # 4. ОТДАЕМ ВИДЕО НАПАРНИКУ (в Base64)
        video_data = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')

        # 5. Чистим за собой только временные файлы
        if os.path.exists(source_path): os.remove(source_path)
        if os.path.exists(output_path): os.remove(output_path)

        # Возвращаем результат серверу напарника
        return {
            "success": True,
            "videoBase64": video_data, # Напарник заберет это и сохранит у себя
            "message": "круто"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})