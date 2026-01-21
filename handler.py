import runpod
import subprocess
import os
import base64
import sys
import urllib.request
import requests

def handler(job):
    try:
        job_input = job["input"]
        
        # Идентификаторы задачи
        requestId = job_input.get("requestId")
        userId = job_input.get("userId")
        callbackUrl = job_input.get("callbackUrl")
        
        print(f"🎬 Processing Request: {requestId}")
        if callbackUrl:
            print(f"📬 Callback URL: {callbackUrl}")

        # 1. Загрузка шаблона (С КЭШИРОВАНИЕМ)
        template_url = job_input.get("templateUrl")
        template_path = job_input.get("templatePath")
        
        if template_url and template_path:
            # Создаем папку, если её нет
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            
            # Проверяем, есть ли файл. Если есть - НЕ качаем заново.
            if not os.path.exists(template_path):
                print(f"⬇️ Downloading template from: {template_url}")
                try:
                    urllib.request.urlretrieve(template_url, template_path)
                    print(f"✅ Template saved to: {template_path}")
                except Exception as e:
                    return {"success": False, "error": f"Failed to download template: {str(e)}"}
            else:
                print(f"⚡ Template found in cache: {template_path}")

        # 2. Сохранение лица (URL или Base64)
        face_url = job_input.get("faceUrl")
        face_base64 = job_input.get("faceBase64")
        face_save_path = job_input.get("facePath")
        
        if face_save_path:
            os.makedirs(os.path.dirname(face_save_path), exist_ok=True)
            try:
                if face_url:
                    print(f"⬇️ Downloading face from: {face_url}")
                    urllib.request.urlretrieve(face_url, face_save_path)
                elif face_base64:
                    if "," in face_base64:
                        face_base64 = face_base64.split(",")[1]
                    with open(face_save_path, "wb") as f:
                        f.write(base64.b64decode(face_base64))
                print(f"✅ Face image saved: {face_save_path}")
            except Exception as e:
                return {"success": False, "error": f"Failed to save face image: {str(e)}"}

        # 3. Запуск команды (FaceFusion)
        args = job_input.get("args")
        if not args:
            return {"success": False, "error": "No args provided"}

        print(f"🚀 Running: python {' '.join(args)}")
        
        # Запускаем из папки /app, где лежит facefusion.py
        result = subprocess.run(["python"] + args, cwd="/app", capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            if callbackUrl:
                try:
                    requests.post(callbackUrl, json={
                        "requestId": requestId,
                        "userId": userId,
                        "success": False,
                        "error": result.stderr
                    }, timeout=10)
                except: pass
            return {"success": False, "error": result.stderr, "stdout": result.stdout}

        # 4. Обработка результата
        output_path = job_input.get("outputPath")
        video_data = None
        
        if output_path and os.path.exists(output_path):
            print(f"✅ Output found: {output_path}")
            with open(output_path, "rb") as v:
                video_data = base64.b64encode(v.read()).decode('utf-8')
            
            # Отправка на сервер напарника (Callback)
            if callbackUrl:
                print(f"📡 Sending callback to {callbackUrl}")
                try:
                    r = requests.post(callbackUrl, json={
                        "requestId": requestId,
                        "userId": userId,
                        "success": True,
                        "videoBase64": video_data
                    }, timeout=60) # Увеличил таймаут для больших видео
                    print(f"Callback response: {r.status_code}")
                except Exception as cb_err:
                    print(f"⚠️ Callback failed: {cb_err}")

        # 5. Очистка (ВАЖНО: template_path НЕ удаляем!)
        try:
            if face_save_path and os.path.exists(face_save_path): os.remove(face_save_path)
            # if template_path and os.path.exists(template_path): os.remove(template_path) <--- ЗАКОММЕНТИРОВАНО ДЛЯ КЭША
            if output_path and os.path.exists(output_path): os.remove(output_path)
        except: pass

        return {
            "success": True,
            "videoBase64": video_data if not callbackUrl else None, 
            "message": "Render complete"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})