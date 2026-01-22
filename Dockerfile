# Базовый образ с CUDA 11.8 и PyTorch для стабильной работы с RTX картами
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Установка системных зависимостей + cuDNN
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libgomp1 \
    python3-tk \
    wget \
    git \
    libcudnn8=8.6.0.163-1+cuda11.8 \
    libcudnn8-dev=8.6.0.163-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Клонирование FaceFusion версии 3.0.0 (стабильная)
RUN git clone --branch 3.0.0 --depth 1 https://github.com/facefusion/facefusion.git . && \
    ls -la && \
    echo "Содержимое директории после клонирования:"

# КРИТИЧЕСКИ ВАЖНО: Удаляем все версии numpy и onnxruntime перед установкой
RUN pip uninstall -y numpy onnxruntime onnxruntime-gpu 2>/dev/null || true

# Устанавливаем СТРОГО numpy 1.26.4 (НЕ 2.x!)
RUN pip install --no-cache-dir "numpy==1.26.4"

# Устанавливаем onnxruntime-gpu строго 1.17.1 для CUDA 11.8
RUN pip install --no-cache-dir "onnxruntime-gpu==1.17.1"

# Устанавливаем основные зависимости с фиксацией версий (БЕЗ обновления numpy и onnxruntime)
RUN pip install --no-cache-dir \
    "opencv-python>=4.8.0,<5.0.0" \
    "pillow>=10.0.0,<11.0.0" \
    "tqdm>=4.66.0" \
    "requests>=2.31.0" \
    "insightface>=0.7.3" \
    "onnx>=1.15.0,<2.0.0" \
    "filetype" \
    "pyyaml" \
    "protobuf" \
    "gdown" \
    "inquirer" \
    "gradio"

# ВАЖНО: Не используем requirements.txt и install.py, так как они переустанавливают onnxruntime!
# Вместо этого устанавливаем только то что нужно FaceFusion

# Установка RunPod SDK для работы с serverless
RUN pip install --no-cache-dir runpod

# Создание директории для моделей
RUN mkdir -p /root/.facefusion/models

# Предзагрузка критически важных моделей для избежания ошибок валидации


# Модели будут скачаны автоматически FaceFusion при первом запуске
# Это надежнее чем пытаться скачать их вручную при сборке образа

# ФИНАЛЬНАЯ ПРОВЕРКА И БЛОКИРОВКА ВЕРСИЙ
# Принудительно переустанавливаем правильные версии если что-то их изменило
RUN pip uninstall -y numpy onnxruntime onnxruntime-gpu 2>/dev/null || true && \
    pip install --no-cache-dir "numpy==1.26.4" "onnxruntime-gpu==1.17.1"

# Проверка установленных версий
RUN python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}'); print(f'Providers: {onnxruntime.get_available_providers()}')"

# Копирование обработчика
COPY handler.py /app/handler.py

# Настройка переменных окружения для корректной работы CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# Запуск handler при старте контейнера
CMD ["python", "-u", "handler.py"]