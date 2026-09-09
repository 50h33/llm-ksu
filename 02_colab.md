# Lab 1 – 코딩 플랫폼 구축 (Colab)

## 파이썬 개발에 VS Code를 쓰는 이유는?

- **스마트 엔진 (Pylance)**: 실시간 오류 검출과 문맥을 인식하는 코드 자동완성.
- **하이브리드 워크플로우**: Jupyter Notebook과 파이썬 스크립트를 하나의 창에서 실행.
- **환경 동기화**: 가상 환경(Virtual Environment)의 자동 감지 및 원클릭 전환.
- **올인원 콘솔**: 통합 터미널과 디버거로 앱을 옮겨 다니지 않고 로직 수정.
- **AI 준비 완료**: LangGraph 에이전트 구축에 필수인 Docker와 Git 기본 지원.

Visual Studio Code를 https://code.visualstudio.com/download 에서 내려받아 컴퓨터에 설치할 것

---

## 가상 환경 생성 (Create a Virtual Environment)

- Windows, VS Code, NVIDIA RTX 4060 Ti (8GB) 기준으로 독립적 실행환경 구축
- Conda 환경은 서로 다른 python version과 package를 독립적으로 관리

```
Windows
├─ NVIDIA 그래픽 드라이버
├─ Anaconda
├─ VS Code
│  ├─ Python 확장
│  └─ Jupyter 확장
└─ Conda 환경: llm
   ├─ Python 3.11
   ├─ PyTorch + CUDA runtime
   ├─ Transformers
   ├─ Datasets
   ├─ Accelerate
   ├─ PEFT / TRL
   ├─ LangChain / LangGraph
   └─ Jupyter kernel
```

---

## NVIDIA 드라이버 확인 (NVIDIA Driver Check)

**Windows PowerShell에서**

NVIDIA 시스템 관리 인터페이스 (NVIDIA system management interface)

```
nvidia-smi
```

---

## NVIDIA 드라이버 설치 (드라이버가 없는 경우)

**Windows PowerShell에서**

NVIDIA 사이트 https://www.nvidia.com/en-us/drivers/details/274317/ 방문

---

## Anaconda 설치 (Anaconda Installation)

**다운로드 및 설치**

이 사이트에서 (https://www.anaconda.com/download/success)

**설치 확인**

Anaconda PowerShell Prompt에서

```
conda --version
```

**Conda 자체 업데이트**

```
conda update -n base -c defaults conda
```

**기존 환경 목록 확인**

```
conda env list
```

---

## CUDA 설치 (본 실습에서는 생략)

**Windows PowerShell에서**

CUDA (Compute Unified Device Architecture)

NVIDIA의 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델

프로젝트 특성을 반영하여 가상 환경 안에 설치할 예정.

---

## 프로젝트 폴더 생성 (Create a project folder)

**새 디렉터리 생성 (Windows에서)**

```
llm_lab01
```

**VS Code에서 열기 (VS Code에서) - 나중에 진행**

File → Open Folder → llm_lab01 선택

---

## 가상 환경 생성 (Create a Virtual Environment)

Python 3.11, NVIDIA graphic driver, Git, PyTorch, Transformers, LangChain
을 포함하는 독립적 개발환경 구축이 최종 목표

**가상 환경 생성**

```
conda create –n llm_lab01 python=3.11 pip -y
```

**생성한 가상 환경 활성화**

```
conda activate llm_lab01
```

환경이 (base)에서 (llm_lab01)로 변경됨

---

## pip 기본 도구 업그레이드 (pip basic tool upgrade)

**Anaconda PowerShell Prompt에서 (llm_lab01 env)**

**pip 도구 업그레이드**

```
python -m pip install --upgrade pip setuptools wheel
```

**패키지를 설치할 때는 아래 형식을 따를 것**

```
python -m pip install package_name
```

---

## Pytorch GPU 버전 설치 (Pytorch GPU version Installation)

**Anaconda PowerShell Prompt에서 (llm_lab01 env)**

**Pytorch 사이트 pytorch.org 방문**

설치에 맞는 명령어 찾기

> **참고:** 최신 안정 버전 PyTorch는 Python 3.10 이상이 필요합니다.
>
> | 항목 | 선택 |
> | --- | --- |
> | PyTorch Build | Stable (2.13.0) |
> | Your OS | Windows |
> | Package | Pip |
> | Language | Python |
> | Compute Platform | CUDA 12.6 |
> | Run this Command | `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |

```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

- `python -m pip install`:
  - 현재 활성화된 파이썬 환경을 사용해 pip 패키지 관리자 모듈을 실행하라는 의미다.
  - 그냥 `pip`를 입력하는 것보다 `-m pip`를 쓰는 방식이 일반적으로 권장되는데, 지금 실행 중인 정확한 파이썬 환경에 패키지가 설치되도록 보장해 주기 때문이다.
- `torch torchvision`: 설치되는 패키지들 — PyTorch와 그 동반 컴퓨터 비전 라이브러리.
- `--index-url [https://download.pytorch.org/whl/cu126]` (https://download.pytorch.org/whl/cu126):
  - 가장 핵심적인 플래그다. 기본적으로 pip는 공개 Python Package Index(PyPI)에서 패키지를 내려받는다. 이 플래그는 그 기본 동작을 덮어써서, CUDA 12.6(cu126)용으로 특별히 컴파일된 PyTorch 공식 wheel 저장소에서 PyTorch를 받도록 지시한다.
  - 이를 통해 기본값인 CPU 전용 버전이 아니라 GPU 가속이 적용된 PyTorch를 받게 된다.

---

**Pytorch GPU 동작 확인**

```
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Pytorch GPU Memory 확인**

```
python -c "import torch; p=torch.cuda.get_device_properties(0); print(round(p.total_memory/1024**3, 2), 'GB')"
```

---

## Hugging Face 기본 패키지 설치 (Hugging Face basic packages installation)

**Conda 자체 업데이트 (Update the Conda itself)**

```
python -m pip install transformers datasets accelerate evaluate tokenizers sentencepiece safetensors huggingface-hub
```

---

## VS Code와 Conda 환경 연결 (Connecting VS Code to Conda Environments)

**확장(Extension) 설치**

- Python - Microsoft
- Python Debugger - Microsoft
- Jupyter - Microsoft
- Python Environments - Microsoft

---

**프로젝트 폴더 열기**

File → Open Folder → llm_lab01 폴더 선택

**파이썬 인터프리터 선택**

ctl + shift + P → Python: Select Interpreter

llm_lab01 환경 선택
(C:\Users\sukang_smu\.conda\envs\llm_lab01 에 실제 존재)

---

## VS Code 사용하기 (Using VS Code)

VS Code 터미널로 환경 확인하기

**VS Code 내부에서 터미널 열기**

Terminal → New Terminal

**터미널에서 환경 확인**

프롬프트에 (llm_lab01) 표시

---

현재 인터프리터 확인

**터미널에서 명령 실행**

```
python -c "import sys; print(sys.executable)"
```

---

**Anaconda PowerShell에서** Jupyter Notebook 커널 등록

**Jupyter 및 커널 패키지 설치**

```
python -m pip install jupyter ipykernel
```

**커널 등록**

```
python -m ipykernel install --user --name llm_lab01 --display-name "Python (llm_lab01)"
```

- Markdown cell
- Code cell
- Server

---

## 첫 파이썬 파일 만들기 (Create your first Python file)

**생성**

main.py

**코드 작성**

```python
Print("Hello Python Project")
```

**실행**

Run (▶ 누르기)

---

## LLM 기본 패키지 설치 (LLM Basic package Installation)

**Hugging Face 기반 LLM 기본 패키지**

이 명령어는 (Qwen 같은) 최신 머신러닝 모델을 파이썬에서 내려받고, 실행하고, 처리하고, 평가하는 데 필요한 Hugging Face AI 핵심 생태계를 설치한다.

```
python -m pip install transformers datasets accelerate evaluate sentencepiece safetensors huggingface-hub tokenizers
```

---

- **transformers**: 사전학습된 AI 모델(LLM, 비전, 오디오)을 불러오기 위한 메인 프레임워크.
- **datasets**: 학습 및 테스트용 공개 AI 데이터셋 수천 종에 손쉽게 접근하게 해 준다.
- **accelerate**: GPU 전반에서 모델 로딩과 성능을 최적화한다(VRAM 관리 및 멀티 GPU 구성을 처리).
- **evaluate**: 모델 성능과 정확도 지표를 측정하는 도구.
- **sentencepiece**: LLaMA, T5 같은 모델이 텍스트를 서브워드로 토크나이즈하기 위해 요구하는 텍스트 처리 라이브러리.
- **safetensors**: 보안 위험 없이 모델 가중치 파일을 저장하고 불러오는 데 사용되는 빠르고 안전한 파일 형식.
- **huggingface-hub**: Hugging Face 클라우드에서 모델과 데이터셋을 프로그램적으로 내려받는 유틸리티.
- **tokenizers**: transformers 라이브러리를 위해 원시 텍스트 처리를 담당하는 Rust 기반 고속 토크나이저.

---

## 전체 환경 확인 (Check the Overall Environment)

지금까지 구성한 모든 환경 확인

check_env.py 작성

```python
import platform
import sys
import accelerate
import datasets
import torch
import transformers

def main() -> None:
    print("=" * 60)
    print("LLM Development Environment")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Operating system : {platform.platform()}")
    print(f"PyTorch : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"Datasets : {datasets.__version__}")
    print(f"Accelerate : {accelerate.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA : {torch.version.cuda}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        memory_gb = properties.total_memory / 1024**3
        print(f"GPU : {gpu_name}")
        print(f"GPU memory : {memory_gb:.2f} GB")
        x = torch.rand((1024, 1024), device="cuda")
        y = x @ x
        print(f"GPU tensor test : OK, {tuple(y.shape)}")
    else:
        print("GPU tensor test : FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

## 파인튜닝 패키지 설치 (Fine-tuning Package Installation)

**LoRA 및 instruction 파인튜닝용**

```
python -m pip install peft trl
```

**학습 로깅 및 평가용**

```
python -m pip install tensorboard scikit-learn pandas matplotlib tqdm
```

**QLoRA용 (4 bit, 8 bit 양자화)**

```
python -m pip install bitsandbytes
```

---

## LLM 기본 패키지 설치 (LLM Basic package Installation)

**Hugging Face 기반 LLM 기본 패키지**

- **transformers**: 사전학습된 AI 모델(LLM, 비전, 오디오)을 불러오기 위한 메인 프레임워크.
- **datasets**: 학습 및 테스트용 공개 AI 데이터셋 수천 종에 손쉽게 접근하게 해 준다.
- **accelerate**: GPU 전반에서 모델 로딩과 성능을 최적화한다(VRAM 관리 및 멀티 GPU 구성을 처리).
- **evaluate**: 모델 성능과 정확도 지표를 측정하는 도구.
- **sentencepiece**: LLaMA, T5 같은 모델이 텍스트를 서브워드로 토크나이즈하기 위해 요구하는 텍스트 처리 라이브러리.
- **safetensors**: 보안 위험 없이 모델 가중치 파일을 저장하고 불러오는 데 사용되는 빠르고 안전한 파일 형식.
- **huggingface-hub**: Hugging Face 클라우드에서 모델과 데이터셋을 프로그램적으로 내려받는 유틸리티.
- **tokenizers**: transformers 라이브러리를 위해 원시 텍스트 처리를 담당하는 Rust 기반 고속 토크나이저.