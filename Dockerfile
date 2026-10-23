FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIG_ID_DATA_DIR=/data

ARG INSTALL_CV=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 sigid \
    && useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home sigid \
    && mkdir /data && chown sigid:sigid /data

WORKDIR /app
COPY requirements/build.txt requirements/runtime.txt /tmp/requirements/
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements/build.txt \
    && pip install --no-cache-dir --require-hashes -r /tmp/requirements/runtime.txt
COPY pyproject.toml ./
COPY sigid ./sigid
RUN if [ "$INSTALL_CV" = "1" ]; then \
        pip install --no-cache-dir --no-build-isolation '.[cv]'; \
    else \
        pip install --no-cache-dir --no-build-isolation --no-deps .; \
    fi \
    && rm -rf /app /tmp/requirements

WORKDIR /data
USER 10001:10001
VOLUME ["/data"]
CMD ["sig-id-serve"]
