FROM python:3.10-slim

# 1. تثبيت مكتبات النظام المطلوب معالجة الصور بها (OpenCV / PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. إعداد مستخدم عادي لمتطلبات Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HF_HOME="/tmp/huggingface"
# إضافة مجلد src إلى مسار البايثون لتسهيل استيراد الـ modules
ENV PYTHONPATH="/app/src:$PYTHONPATH"

WORKDIR /app

# 3. نسخ الاعتماديات وتثبيتها
COPY --chown=user ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 4. نسخ بقية المشروع
COPY --chown=user . /app

# 5. فتح المنفذ الخاص بـ HF Spaces
EXPOSE 7860

# 6. تشغيل التطبيق من داخل مجلد src
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]