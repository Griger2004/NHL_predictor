FROM python:3.12-slim
WORKDIR /app
COPY main/backend/requirements.txt ./requirements.txt
RUN python -c "import pathlib; p=pathlib.Path('requirements.txt'); r=p.read_bytes(); p.write_text(r.decode('utf-16') if r[:2] in (b'\xff\xfe',b'\xfe\xff') else r.decode('utf-8'), encoding='utf-8')"
RUN pip install --no-cache-dir -r requirements.txt xgboost==3.1.2
COPY main/backend/ ./backend/
COPY main/ml_dev/scripts/ ./ml_dev/scripts/
COPY main/ml_dev/notebooks/models/ ./ml_dev/notebooks/models/
EXPOSE 5000
CMD ["python", "backend/app.py"]
