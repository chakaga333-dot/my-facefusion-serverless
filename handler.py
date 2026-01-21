import runpod
import subprocess
import os
import base64
import sys
import urllib.request
import requests

# Твоя диагностика CUDA остается (важно для логов)
try:
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    print(f"✅ CUDA статус: {'CUDAExecutionProvider' in providers}")
except:
    print("❌ Ошибка диагностики")

def handler(job):
    try:
        job_input = job["input"]
        requestId = job_input.get("requestId", "task")
        callbackUrl = job_input.get("callbackUrl")

        # 1. Подготовка путей (как в твоем рабочем коде)
        os.makedirs("/tmp/input", exist_ok=True)
        os.makedirs("/tmp/output", exist_ok=True)
        
        source_path = "/tmp/input/source.jpg"
        target_path = job_input.get("templatePath", "/tmp/input/target.mp4")
        output_path = job_input.get("outputPath", "/tmp/output/result.mp4")

        # 2. Сохранение лица (URL или Base64 от напарника)
        face_url = job_input.get("faceUrl")
        face_base64 = job_input.get("faceBase64")
        
        if face_url:
            urllib.request.urlretrieve(face_url, source_path)
        elif face_base64:
            if "," in face_base64: face_base64 = face_base64.split(",")[1]
            with open(source_path, "wb") as f:
                f.write(base64.b64decode(face_base64))

        # 3. Шаблон (Скачивание, если его нет в /runpod-volume)
        template_url = job_input.get("templateUrl")
        if template_url and not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            urllib.request.urlretrieve(template_url, target_path)

        # 4. ФОРМИРОВАНИЕ КОМАНДЫ (Твоя идеальная база + его надстройка)
        # Если напарник прислал готовые args — берем их, если нет — твой конфиг
        args = job_input.get("args")
        if not args:
            args = [
                "facefusion.py", "headless-run",
                "-s", source_path,
                "-t", target_path,
                "-o", output_path,
                "--processors", "face_swapper",
                "--execution-providers", "cuda",
                "--skip-download" # Чтобы не лез за моделями в сеть
            ]

        print(f"🚀 Запуск: python {' '.join(args)}")
        
        # Твой проверенный запуск
        result = subprocess.run(
            ["python"] + args, 
            cwd="/app", 
            capture_output=True, 
            text=True, 
            timeout=600
        )

        # 5. Обработка результата
        video_data = None
        if os.path.exists(output_path):
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')

        # Колбэк напарнику
        if callbackUrl and video_data:
            try:
                requests.post(callbackUrl, json={
                    "requestId": requestId, 
                    "success": True, 
                    "videoBase64": video_data
                }, timeout=30)
            except: pass

        # Очистка (как в коде напарника) [cite: 15, 31]
        try:
            if os.path.exists(source_path): os.remove(source_path)
            if "/tmp/" in output_path and os.path.exists(output_path): os.remove(output_path)
        except: pass

        return {
            "success": True, 
            "videoBase64": video_data if not callbackUrl else "Sent to Callback",
            "message": "круто" 
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})