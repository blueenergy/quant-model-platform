FROM python:3.11-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY quant_model_platform ./quant_model_platform

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir .

RUN useradd -u 1000 -g 1000 -m -d /tmp appuser && chown -R appuser:appuser /app
USER 1000

ENTRYPOINT ["python", "-m", "quant_model_platform.workers.run_loop"]
CMD ["--limit-per-iteration", "1"]
