FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY templates ./templates
COPY model ./model
EXPOSE 8000
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2"]
