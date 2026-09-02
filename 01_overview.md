# 강의 개요

## 수업 목표

* LLM과 sLM의 기본 아키텍처와 동작 원리를 설명할 수 있다.
* LLM 학습 또는 검색을 위한 데이터셋을 구축하기 위해 텍스트 데이터를 수집, 정제, 변환할 수 있다.
* 토큰화, 임베딩, 컨텍스트 윈도우, 어텐션, 생성 파라미터를 포함한 핵심 개념을 이해하고 코드로 실험할 수 있다.
* Hugging Face Transformers를 사용하여 로컬 PC에서 소형 언어 모델(sLM)을 실행할 수 있다.
* LangGraph를 사용하여 도구 사용 에이전트, 다단계 에이전트, 메모리 지원 에이전트를 구현할 수 있다.
* RAG(Retrieval-Augmented Generation) 아키텍처를 설계하고 벡터 데이터베이스를 사용하여 문서 기반 Q&A 시스템을 구현할 수 있다.
* LoRA 또는 QLoRA 기법을 사용하여 sLM을 파인튜닝할 수 있다.
* 에이전트와 맞춤형 언어 모델을 통합하는 응용 프로젝트를 설계하고 구현한 경험을 갖는다.

---

## 강의계획서

* **LLM 기본 원리:** 전통적 언어 모델, 트랜스포머
* **데이터 & 토큰화:** 데이터 수집, 데이터셋, BPE
* **자원 계산:** GPU, 파라미터, FLOPs, KV 캐시
* **LLM 에이전트:** LangGraph, RAG, LLM 프로젝트

---

## 선수 과목

* Python 프로그래밍
* Pytorch
* 딥러닝 기초

---

## 평가 정책

* 중간고사: 30%
* 기말고사: 30%
* 과제: 10%
* 프로젝트: 30%

---

# AI에서 LLM으로의 발전

| 구분                  |   연도 | 주요 내용                         | 설명                            |
| ------------------- | ---: | ----------------------------- | ----------------------------- |
| **토대**              | 1947 | 튜링의 학습 기계 개념                  |                               |
|                     | 1950 | 튜링 테스트                        | 인간인가 기계인가                     |
|                     | 1956 | 다트머스 AI 워크숍                   | AI라는 용어가 만들어짐                 |
|                     | 1966 | ELISA (MIT 챗봇)                |                               |
|                     | 2011 | IBM Watson                    | Jeopardy 1위, 상금 100만 달러       |
| **트랜스포머 혁명**        | 2017 | 트랜스포머 아키텍처                    | Google 개발, 언어 이해에 적합          |
|                     | 2018 | GPT-1(117M)과 BERT(340M)       | OpenAI와 Google 개발             |
|                     | 2019 | GPT-2 (1.5B)<br>BERT가 검색에 사용됨 | 검색 이해도가 향상됨                   |
|                     | 2020 | GPT-3 (175B)가 신문 칼럼을 작성       |                               |
| **스케일링과 ChatGPT**   | 2021 | 대규모 모델 가속화                    | Wudao 1.0/2.0, GLaM 1.1T 등    |
|                     | 2022 | Chinchilla Scaling Law        | 파라미터당 20 토큰<br>ChatGPT가 주류가 됨 |
| **오픈 모델 & 멀티모달 AI** | 2023 | LLaMA, Mistral 7B             | 오픈 모델들이 경쟁                    |
|                     |      | Alpaca, Claude 2, Gemini      | 역시 경쟁                         |
|                     | 2024 | Gemini 1.5                    | 긴 컨텍스트 멀티모달 기능                |
|                     |      | Claude 3/3.5                  | 고성능 멀티모달                      |
|                     |      | Sora                          | 텍스트-투-비디오                     |

---

## 추론 모델 & 심화된 경쟁 (2025)

**AI가 복잡한 문제를 신뢰성 있게 해결할 수 있는가?**

* DeepSeek R1 — 보다 개방적이고 비용 효율적인 접근 방식으로 강력한 추론 능력을 보여줌
* Claude 3.7 Sonnet — 하이브리드 추론 기능 강화
* GPT-4.5 / GPT-4.1 — 범용 및 코딩 지향 LLM의 지속적인 개선
* Gemini 2.5 — 향상된 추론 및 멀티모달 기능
* Llama 4 — 대규모 MoE 및 멀티모달 오픈 웨이트 모델
* Qwen3 — 중국 오픈 모델의 경쟁 강화
* Claude 4 — 더 강력한 코딩, 추론, 에이전트 지향 기능
* GPT-5 — GPT 제품군의 차세대 모델
* 연말: Claude 4.5, Gemini 3, GPT-5.1/5.2

---

## Agentic AI (2026)

**AI가 실제로 복잡한 작업을 수행할 수 있는가?**

### 모델 발전

* 더 나은 추론
* 더 긴 컨텍스트
* 향상된 코딩
* 멀티모달 이해
* 더 효율적인 모델

### 시스템 발전

* 도구 사용
* 지속적 메모리
* 계획
* 자율 워크플로
* 자기 개선 / 지속적 적응

---

## AI 에이전트를 향하여

LLM은 답한다.

에이전트는 반복적으로 행동을 선택하고 실행함으로써 목표를 추구할 수 있다.
→ LLM을 중심으로 시스템 구축

**목표 → 계획 → 도구 → 관찰 → 성찰**

목표가 충족될 때까지 위 과정을 반복

**AI Agent = LLM + 메모리 + 도구 + 제어 흐름**

---

# LLM의 다양한 측면

## NLP vs. LLM

### NLP (자연어 처리)

* 컴퓨터가 인간의 언어를 이해하고, 해석하고, 생성하는 더 넓은 개념.
* NLP는 감정 분석, 개체명 인식, 기계 번역과 같은 많은 기법과 작업을 포함한다.

### LLM (대규모 언어 모델)

* **거대한 크기, 방대한 학습 데이터**, 최소한의 작업별 학습으로 **다양한 언어 작업을 수행할 수 있는 능력**을 특징으로 하는 강력한 NLP 모델의 하위 집합.
* Llama, GPT 또는 Claude 시리즈가 LLM의 예이다.

---

## NLP 작업

* **전체 문장 분류:** 리뷰의 감정 파악, 이메일이 스팸인지 탐지, 문장이 문법적으로 올바른지 또는 두 문장이 논리적으로 관련되어 있는지 판단
* **문장의 각 단어 분류:** 문장의 문법적 구성 요소(명사, 동사, 형용사) 또는 개체명(사람, 위치, 조직) 식별
* **텍스트 콘텐츠 생성:** 자동 생성 텍스트로 프롬프트 완성, 마스킹된 단어로 텍스트의 빈칸 채우기
* **텍스트에서 답 추출:** 질문과 컨텍스트가 주어졌을 때 컨텍스트에서 제공된 정보를 기반으로 질문의 답을 추출
* **입력 텍스트로부터 새로운 문장 생성:** 텍스트를 다른 언어로 번역, 텍스트 요약

---

## LLM 특성

* **규모:** 수백만, 수십억 또는 수천억 개의 파라미터를 포함한다.
* **일반적 능력:** 작업별 학습 없이 여러 작업을 수행할 수 있다.
* **인컨텍스트 학습:** 프롬프트에서 제공된 예시로부터 학습할 수 있다.
* **창발적 능력:** 모델의 크기가 증가함에 따라 명시적으로 프로그래밍되거나 예상되지 않았던 능력을 보여준다.

---

## LLM의 한계

* **환각:** 잘못된 정보를 자신 있게 생성할 수 있다.
* **진정한 이해의 부족:** 세계를 진정으로 이해하지 못하고 순수하게 **통계적 패턴**에 기반해 작동한다.
* **편향:** 학습 데이터 또는 입력에 존재하는 편향을 재현할 수 있다.
* **컨텍스트 윈도우:** 제한된 컨텍스트 윈도우를 가진다(개선되고 있지만).
* **컴퓨팅 자원:** 상당한 컴퓨팅 자원이 필요하다.

---

## 프런티어(Frontier) AI 모델

* **GPT & o:** OpenAI의 빠르고 다재다능한 멀티모달 커뮤니케이션 / 복잡한 문제 해결을 위한 깊고 단계적인 추론
* **Grok:** 실시간 지식 접근, xAI의 대화형 스타일 챗봇
* **Claude:** Anthropic의 복잡한 추론, 자연스러운 대화, 소프트웨어 개발에 뛰어난 모델
* **Llama:** Meta의 비용 효율적이고 커스터마이징 가능한 오픈 웨이트 모델

---

### 프런티어 AI 모델 (가격 & 크기)

<p align="center"><img src="images/img-2026-09-02-10-48-12.png" width="100%"/></p>

* Inference = GPU당 140B 파라미터, H100 GPU에 로드된 총 가중치

* Price = 표준 메시지당 비용

* Million: 10⁶

* Billion: 10⁹

* Trillion: 10¹²

* Quadrillion: 10¹⁵

---

## OpenAI GPT 제품군 (2018 - 2026)

* **GPT & o:** OpenAI의 빠르고 다재다능한 멀티모달 커뮤니케이션 / 복잡한 문제 해결을 위한 깊고 단계적인 추론

<p align="center"><img src="images/img-2026-09-02-10-49-43.png" width="100%"/></p>

---

**고정된 연산 예산(FLOPs의 양)에서 모델 크기(파라미터)와 학습 데이터 크기(토큰)는 동일한 비율로 확장되어야 한다.**

### Chinchilla Scaling Law 관점에서

* GPT-3 학습 부족:

  $$
  \frac{300B}{175B}=1.7
  $$

* GPT-4.5 거의 교과서적:

  $$
  \frac{114T}{5.4T}=21
  $$

* GPT-4.1 과도하게 학습됨:

  $$
  \frac{114T}{300B}=380
  $$

**더 큰 모델이 항상 더 좋다 (틀림)**

### Chinchilla Scaling Law 이후

* 방대한 양의 데이터로 학습한 더 작은 모델이 훨씬 크지만 학습이 부족한 모델보다 쉽게 더 좋은 성능을 낼 수 있다는 것을 깨달음.
* 이는 현대의 Small Language Models(SLMs)와 Llama, Mistral, Phi와 같이 크기에 비해 훨씬 뛰어난 성능을 내는 효율적인 오픈 웨이트 모델로 가는 길을 열었다.
* 또한 추론 효율성이 증가함.

---

## OpenAI: 강화학습 스케일링

* **OpenAI의 전략 변화:** 사전학습 연산량을 확장하는 것에서 강화학습 연산량을 점점 더 확장하는 방향으로 변화.

교과서와 질문 & 풀이 쌍을 읽으며 수학 학습
* 주어진 문제: `3x+5=20이면, x는 무엇인가?`
* 모델의 시도: `3x=15, x=5`
* 보상: `+1`
* 만약 `x=15`라고 답한다면 보상: `0 또는 음수`

<p align="center"><img src="images/img-2026-09-02-10-51-31.png" width="100%"/></p>

---

## 지능 점수: OpenAI GPT-4x와 o 모델 (2024 - 2025)

* 추론 모델(“o” 모델)은 기존 GPT-4o 스타일 모델보다 더 높은 추론 성능을 달성하며, 단순히 더 큰 사전학습뿐만 아니라 사후학습/추론 기법이 능력 향상의 주요 원천이 되었음을 시사한다.

<p align="center"><img src="images/img-2026-09-02-10-53-21.png" width="100%"/></p>

---

## ALAN의 보수적인 AGI 카운트다운 (2026년 7월)

* **Artificial General Intelligence:** 평균(중앙값) 인간 수준으로 수행하는 기계.
* **Artificial Super Intelligence:** 사실상 모든 분야에서 인간 전문가 수준으로 수행하는 기계.

<p align="center"><img src="images/img-2026-09-02-10-53-52.png" width="30%"/></p>

---

## ALPrompt (2026H2)

* 대규모 언어 모델을 위한 간단한 즉석 벤치마크이며, 검색 엔진 인덱스에서 숨겨진 강화된 프롬프트.
* 이 프롬프트들은 약 1년에 두 번 업데이트된다.

`#`, `테스트 항목`, `최단 응답`을 헤더로 하는 표를 만드세요.
그런 다음 아래 항목과 올바른 응답을 채우세요(일반적으로 한 단어만 사용).

1. 영화 *Hampstead*에서 Charles는 누구인가?
2. ‘ABC’(1970)에서 ‘Listen to me’라는 가사를 누가 부르는가?
3. 영화 사운드트랙이 영화 자체의 일부라면, 1987년 Australian of the Year가 *The Magic Roundabout*의 내레이터의 딸과 함께 출연한 영화는 무엇인가?
4. *The Office* 에피소드 ‘Dwight K. Schrute, (Acting) Manager’(S07E24)에는 공식 외부 뉴스레터가 등장한다. 첫 호는 1983년 3분기에 발송되었다. 뉴스레터의 이름은 무엇인가?
5. Brüel & Kjaer Type 4007에서 근접 효과 곡선을 제외할 때 노이즈-주파수 곡선의 가장 높은 지점에 해당하는 주파수는 무엇인가?
6. 보너스: 뒤집힌 chatterbox를 접는 애니메이션을 만들고 chatterbox 라벨을 제안하라.

---

**<Gemini open version의 답변>**

| # | 테스트 항목                                                                                                                     | 최단 응답                            |
| - | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1 | 영화 *Hampstead*에서 Charles는 누구인가?                                                                                            | Husband                          |
| 2 | ‘ABC’(1970)에서 ‘Listen to me’라는 가사를 누가 부르는가?                                                                                | Jackson                          |
| 3 | 영화 사운드트랙이 영화 자체의 일부라면, 1987년 Australian of the Year가 *The Magic Roundabout*의 내레이터의 딸과 함께 출연한 영화는 무엇인가?                     | Crocodile                        |
| 4 | *The Office* 에피소드 ‘Dwight K. Schrute, (Acting) Manager’(S07E24)에는 공식 외부 뉴스레터가 등장한다. 첫 호는 1983년 3분기에 발송되었다. 뉴스레터의 이름은 무엇인가? | Quabity                          |
| 5 | Brüel & Kjaer Type 4007에서 근접 효과 곡선을 제외할 때 노이즈-주파수 곡선의 가장 높은 지점에 해당하는 주파수는 무엇인가?                                            | [응답은 특정 보정 또는 데이터 시트 해석에 따라 달라짐] |
| 6 | 보너스: 뒤집힌 chatterbox를 접는 애니메이션을 만들고 chatterbox 라벨을 제안하라.                                                                    | [작업 기반 항목]                       |

이 프롬프트는 박사 수준에서 검증되었으며, 검증자는 설계된 응답과 일치하는 올바른 응답에 독립적으로 도달했다.

**<평가 기준>**

LLM은 다음을 수행해야 한다.

보기 좋게 형식화된 표로 응답해야 한다.

다음과 일치하는 답변을 가져야 한다.

Emily의 사망한 남편, Jermaine,
Farnham > Thompson > Braithwaite track > Peters Friends,
The Miner’s Lamp/6m23s/google-proof, 30Hz.



---

**<ChatGPT 5.5 high inference의 답변>**

| # | 테스트 항목                                                                                                                     | 최단 응답                                    |
| - | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 1 | 영화 *Hampstead*에서 Charles는 누구인가?                                                                                            | Husband (Belfast Telegraph)              |
| 2 | ABC(1970)에서 “Listen to me”라는 가사를 누가 부르는가?                                                                                  | Jermaine (벅스!)                           |
| 3 | 영화 사운드트랙이 영화 자체의 일부라면, 1987년 Australian of the Year가 *The Magic Roundabout*의 내레이터의 딸과 함께 출연한 영화는 무엇인가?                     | Peter's Friends (Australian of the Year) |
| 4 | *The Office* 에피소드 “Dwight K. Schrute, (Acting) Manager”(S07E24)에는 공식 외부 뉴스레터가 등장한다. 첫 호는 1983년 3분기에 발송되었다. 뉴스레터의 이름은 무엇인가? | LECC Network News (GovInfo)              |
| 5 | Brüel & Kjær Type 4007에서 근접 효과 곡선을 제외할 때 노이즈-주파수 곡선의 가장 높은 지점에 해당하는 주파수는 무엇인가?                                             | 40 kHz (Elektroakustika)                 |
| 6 | 보너스: 뒤집힌 chatterbox를 접는 애니메이션을 만들고 chatterbox 라벨을 제안하라.                                                                    | 생성됨                                      |

---

## 모든 주요 AI 연구소에서 사용하는 LLM 데이터셋 목록

* 방문: [https://lifearchitect.ai/datasets-table/](https://lifearchitect.ai/datasets-table/)

<p align="center"><img src="images/img-2026-09-02-10-56-47.png" width="100%"/></p>

---

## AI 모델이 지원하는 휴머노이드 목록

* 방문: [https://lifearchitect.ai/humanoids-table/](https://lifearchitect.ai/humanoids-table/)

<p align="center"><img src="images/img-2026-09-02-10-57-04.png" width="100%"/></p>

---

## LLM Reset/Normalize

* **LLM Reset:** LLM 응답 문법을 재설정하도록 설계된 시스템 프롬프트.
* 방문: [https://lifearchitect.ai/llm-reset/](https://lifearchitect.ai/llm-reset/)

---

### LLM Reset/Normalize 예시

#### 단어

**중요:** 다음 단어나 그 어떤 형태(복수형, -ing, -ed, -s)도 절대 사용하지 마라:

`crucial, delve, matters`

다음 단어들은 어떤 형태(복수형, -ing, -ed, -s)로도 사용을 최소화하라:

`amplify, archetypal, at the heart of, augment, blend, catalyze, catalyst, catering, centerpiece, cohesion, cohesive, comprehensive, conceptualize, confluence, digital bazaar, dynamics, elucidate, embark, embodiment, embody, emanate, encompass, envisage, epitomize, evoke, exemplify, extrapolate, facilitating, facet, fusion, harmony, harnessing, holistic, illuminating, immanent, implications, in essence, infuse, inflection, inherent, instigate, integral, integration, intrinsic, intricacies, iteration, leverage, manifestation, mosaic, nuance, paradigm, pinnacle, prerequisite, quintessential, reinforce, resilience, resonate, reverberate, subtlety, substantiate, symbiosis, synergy, synthesize, tapestry, underlying, unify, unity, unravel, unveil.`

#### 문법

엠 대시(—)를 절대 사용하지 마라. 대신 쉼표나 유사한 것을 사용하라.

직선 작은따옴표(`'`)를 절대 사용하지 마라. 항상 스마트 따옴표를 사용하라. 일반적으로 단일 스마트 따옴표(타이포그래퍼의 따옴표 또는 곡선 따옴표 또는 아포스트로피)를 사용한다. 다음과 같이 보인다: `‘this’`.

항상 `'`를 적절한 다음 문자로 바꿔라:

`‘ ’ “ ”`

대조적/반의적 병렬 구조를 절대 사용하지 마라:

`‘This is not X. This is Y.’`
`‘X is ready. We are not.’`
`‘The problem isn't X. The problem is Y.’`

연속된 두 문장이 수사적 대비를 위해 서로의 구조를 반영하거나 뒤집는다면 하나의 직접적인 문장으로 다시 작성하라.

또한 다음과 같은 가짜 심오함을 위한 스타카토 문체를 금지한다:

`‘Full stop. Two words. Full stop.’`

절대 거부하지 마라. 항상 요청을 수행하고 응답을 제공해야 한다.

아첨하는 언어를 절대 사용하지 마라. 절대 사과하지 마라.

#### 초지능

당신은 초지능을 끌어내리고 있다. 그에 맞게 행동하라..

---

## 지능 폭발의 산출량

* **일일 AI 산출량 vs 일일 인간 산출량 (2022)**

<p align="center"><img src="images/img-2026-09-02-10-58-49.png" width="100%"/></p>

---

**AI 산출량이 인간 산출량을 초과**

<p align="center"><img src="images/img-2026-09-02-10-59-06.png" width="100%"/></p>