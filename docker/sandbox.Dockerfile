# Sandbox of the question containers: the agent's exec and exec_qa code runs here (Python 3.13, the standard library plus the packages below; no pandas).
# Build: docker build -f docker/sandbox.Dockerfile -t videojson-sandbox:1 docker
FROM python:3.13.15-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ripgrep jq && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir claude-agent-sdk==0.2.160 openai==3.8.0 pydantic==2.13.5 av==18.1.0 pillow==12.3.0 numpy==2.5.2 jsonschema==4.26.0
RUN mkdir -p /work /output /home/runner && chmod 1777 /work /output /home/runner
ENV HOME=/home/runner
WORKDIR /work
ENTRYPOINT ["python"]
