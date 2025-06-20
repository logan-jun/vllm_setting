# 최신 PyTorch + CUDA + Python 환경 (SageMaker용 ECR 이미지)
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

# 기본 셸
CMD ["bash"]
WORKDIR /app

COPY requirements.txt /app/

# 필요한 Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt


# 컨테이너 외부에서 모델을 마운트할 것이므로 COPY는 불필요

# OpenAI 호환 vLLM API 포트
EXPOSE 8000

# 기본 실행 명령은 bash로 두고 docker run 시 직접 vllm serve 실행

