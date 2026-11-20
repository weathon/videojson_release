# laya-serve (convaiinnovations/laya, the Decision API model) in a container on the vj-laya network, next to the decision proxy.
# Build: docker build -f docker/laya.Dockerfile -t vj-laya:1 docker
FROM vllm/vllm-openai:v0.31.0
RUN pip install --no-cache-dir "laya[serve]==0.4.19"
ENTRYPOINT ["laya-serve"]
