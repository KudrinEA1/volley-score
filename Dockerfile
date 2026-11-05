# Сервер счёта для хостинга (Hugging Face Spaces и любой другой хостинг с Docker).
FROM python:3.11-slim

# Tesseract - запасной путь распознавания цифр; libglib нужен OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face запускает контейнер от пользователя с номером 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "server.py"]
