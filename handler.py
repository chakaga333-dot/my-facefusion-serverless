import runpod
import subprocess
import os
import sys
import urllib.request
import onnxruntime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

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

# Проверка переменных окружения
print("📋 ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '❌ Не установлена')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '❌ Не установлена')}")
print("=" * 60)
sys.stdout.flush()


def send_email_with_attachment(file_path, recipient_email):
    """
    Отправка видео на email через SMTP Gmail
    
    ВАЖНО: Нужно настроить переменные окружения:
    - SMTP_EMAIL: ваш Gmail (например: yourname@gmail.com)
    - SMTP_PASSWORD: пароль приложения Gmail (не обычный пароль!)
    """
    try:
        # Получение credentials из переменных окружения
        smtp_email = os.environ.get('SMTP_EMAIL')
        smtp_password = os.environ.get('SMTP_PASSWORD')
        
        if not smtp_email or not smtp_password:
            print("⚠️ SMTP credentials не настроены. Видео не отправлено.")
            print("   Установите SMTP_EMAIL и SMTP_PASSWORD в RunPod")
            return False
        
        print(f"\n📧 Отправка видео на {recipient_email}...")
        
        # Создание сообщения
        msg = MIMEMultipart()
        msg['From'] = smtp_email
        msg['To'] = recipient_email
        msg['Subject'] = "✅ Ваше FaceFusion видео готово!"
        
        # Текст письма
        body = """
Здравствуйте!

Ваше видео с замененным лицом успешно обработано и готово к просмотру.

Видео прикреплено к этому письму.

---
FaceFusion RunPod Service
        """
        msg.attach(MIMEText(body, 'plain'))
        
        # Прикрепление видео
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"📎 Прикрепляю файл ({file_size_mb:.2f} MB)...")
        
        with open(file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename=facefusion_result.mp4'
        )
        msg.attach(part)
        
        # Отправка через Gmail SMTP
        print("🔄 Подключение к Gmail SMTP...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(smtp_email, smtp_password)
        
        print("📤 Отправка письма...")
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Письмо успешно отправлено на {recipient_email}!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при отправке email: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def download_file(url, output_path):
    """
    Скачивание файла по URL с отображением прогресса
    """
    try:
        print(f"📥 Скачиваю файл: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Файл сохранен: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Ошибка при скачивании: {str(e)}")
        raise


def process_facefusion(job):
    """
    Основной обработчик задачи FaceFusion
    
    Ожидаемые параметры в job['input']:
    - source: URL фотографии источника (лицо для замены)
    - target: URL видео цели (куда вставляем лицо)
    
    Возвращает:
    - success: True/False
    - output_path: путь к результату (если успешно)
    - error: описание ошибки (если провал)
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 НАЧАЛО ОБРАБОТКИ ЗАДАЧИ")
        print("=" * 60)
        
        job_input = job["input"]
        source_url = job_input.get("source")
        target_url = job_input.get("target")
        
        # Валидация входных данных
        if not source_url or not target_url:
            error_msg = "❌ Не указаны обязательные параметры 'source' или 'target'"
            print(error_msg)
            return {"error": error_msg}
        
        print(f"📸 Source URL: {source_url}")
        print(f"🎬 Target URL: {target_url}")
        
        # Создание временных директорий
        os.makedirs("/tmp/input", exist_ok=True)
        os.makedirs("/tmp/output", exist_ok=True)
        
        # Определение путей к файлам
        source_path = "/tmp/input/source.jpg"
        target_path = "/tmp/input/target.mp4"
        output_path = "/tmp/output/result.mp4"
        
        # Скачивание исходных файлов
        print("\n📥 СКАЧИВАНИЕ ФАЙЛОВ:")
        download_file(source_url, source_path)
        download_file(target_url, target_path)
        
        # Формирование команды для запуска FaceFusion
        # ВАЖНО: Используем facefusion.py, а не run.py!
        command = [
            "python", "facefusion.py",
            "headless-run",
            "-s", source_path,                # Source (короткая форма)
            "-t", target_path,                # Target (короткая форма)
            "-o", output_path,                # Output (короткая форма)
            "--processors", "face_swapper",   # Только замена лиц
            "--execution-providers", "cuda",  # ОБЯЗАТЕЛЬНО GPU
            "--execution-thread-count", "4",  # 4 потока для GPU
            "--execution-queue-count", "2",   # Очередь для параллелизма
            "--video-memory-strategy", "moderate",  # Умеренное использование памяти
            "--face-detector-model", "yoloface",    # Быстрая модель детекции
            "--face-detector-size", "640x640"
        ]
        
        print("\n🔧 КОМАНДА ЗАПУСКА:")
        print(" ".join(command))
        print("\n⏳ Обработка началась (макс. 10 минут)...")
        sys.stdout.flush()
        
        # Запуск процесса FaceFusion с увеличенным таймаутом для первого запуска
        # (может потребоваться время на скачивание моделей)
        result = subprocess.run(
            command,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=600  # Таймаут 10 минут для первого запуска с загрузкой моделей
        )
        
        # Вывод логов в RunPod
        print("\n📄 STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️ STDERR:")
            print(result.stderr)
        
        sys.stdout.flush()
        
        # Проверка кода возврата
        if result.returncode != 0:
            return {
                "error": "Процесс FaceFusion завершился с ошибкой",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        
        # Проверка создания выходного файла
        if not os.path.exists(output_path):
            return {"error": "Выходной файл не был создан"}
        
        file_size = os.path.getsize(output_path)
        print(f"\n✅ УСПЕХ! Файл создан: {output_path}")
        print(f"📦 Размер файла: {file_size / 1024 / 1024:.2f} MB")
        
        # Отправка результата на email
        recipient_email = job_input.get("email", "chakaga@mail.ru")  # Email по умолчанию
        email_sent = send_email_with_attachment(output_path, recipient_email)
        
        return {
            "success": True,
            "output_path": output_path,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "email_sent": email_sent,
            "recipient_email": recipient_email,
            "message": "Обработка успешно завершена" + (" и отправлена на email" if email_sent else "")
        }
        
    except subprocess.TimeoutExpired:
        error_msg = "⏱️ Превышен таймаут обработки (10 минут)"
        print(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"❌ Неожиданная ошибка: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg}


# ============================================================
# ЗАПУСК RUNPOD SERVERLESS HANDLER
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 ЗАПУСК FACEFUSION RUNPOD HANDLER")
    print("=" * 60)
    sys.stdout.flush()
    
    runpod.serverless.start({"handler": process_facefusion})