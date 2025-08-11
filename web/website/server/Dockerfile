FROM python:3.10-slim

WORKDIR /app

# show what we copy (debug tip)
COPY requirements.txt .
RUN ls -la /app && python -V && pip -V

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
