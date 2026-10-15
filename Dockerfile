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

# Dependencies in their own layer, keyed on pyproject.toml alone. The ACR relay
# pulls this image cross-border at 50 KB/s - 6 MB/s; while the install shared a
# layer with the source, every commit changed that layer's digest and re-sent
# all 185 MB of it. Keep this RUN above the source COPY -- merging the two back
# together costs 185 MB per commit.
COPY pyproject.toml README.md ./
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    python -c 'import tomllib;print(chr(10).join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))' > /tmp/deps.txt && \
    pip install --no-cache-dir -r /tmp/deps.txt && \
    rm /tmp/deps.txt

COPY quant_model_platform ./quant_model_platform
RUN pip install --no-cache-dir --no-deps .

RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m appuser && \
    chown -R appuser:appgroup /app
USER appuser

ENTRYPOINT ["python", "-m", "quant_model_platform.workers.run_loop"]
CMD ["--limit-per-iteration", "1"]
