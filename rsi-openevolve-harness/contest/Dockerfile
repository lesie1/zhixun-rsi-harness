# 智训引擎 · 工业模型自进化 Harness，比赛镜像 v1。
# 构建上下文由 contest/build_context.sh 生成（sdk/ + app/）。镜像里没有密钥、数据和模型权重。
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BASELINE_CPU_THREADS=1
RUN pip install --no-cache-dir numpy==1.26.4
COPY sdk /opt/contest-harness
RUN pip install --no-cache-dir /opt/contest-harness
COPY app /app
WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
