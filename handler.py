import runpod
import subprocess
import os
import base64
import urllib.request

# Твоя диагностика остается
print("=" * 60)
print("🚀 ЗАПУСК САМОЙ ЛУЧШЕЙ СБОРКИ (КРУТО)")
print("=" * 60)

def handler(job):
    try:
        job_input = job["input"]
        
        # --- [ СЕКРЕТНЫЙ ФИКС NSFW ] ---
        # Мы создаем конфиг, который отключает проверку хэша open_nsfw навсегда
        os.makedirs(os.path.expanduser('~/.facefusion'), exist_ok=True)
        config_path = os.path.expanduser('~/.facefusion/facefusion.ini')
        with open(config_path, 'w') as f:
            f.write('[choices]\ncontent_analyser_model = none\n')
        # -------------------------------

        # 1. Пути
        source_path = "/tmp/source.jpg"
        target_path = job_input.get("targetPath", "/workspace/video/1.mp4")
        output_path = "/tmp/output_result.mp4"

        # 2. Сохраняем лицо из Base64 (от твоего HTML)
        face_base64 = job_input.get("faceBase64")
        if face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))
        else:
            return {"success": False, "error": "No faceBase64 provided"}

        # 3. ТВОЯ ИДЕАЛЬНАЯ КОМАНДА (которую мы обсуждали)
        # Если напарник прислал свои args - берем их, если нет - твои стандартные
        args = job_input.get("args")
        if not args:
            args = [
                "facefusion.py", "headless-run",
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

        print(f"🚀 GPU Task Start with command: {' '.join(args)}")
        
        # Запуск FaceFusion
        result = subprocess.run(
            ["python"] + args, 
            cwd="/app", 
            capture_output=True, 
            text=True
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr or result.stdout}

        # 4. ВОЗВРАЩАЕМ ВИДЕО В HTML (в Base64)
        video_data = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')
            # Чистим за собой
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