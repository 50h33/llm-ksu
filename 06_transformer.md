# Transformer

## 단어 임베딩에서 Transformer까지 (2013 - 2017)

## 순환 신경망

### 바닐라 신경망

<p align="center"><img src="images/img-2026-10-07-09-22-13.png" width="10%"/></p>

### 순환 신경망(Recurrent Neural Networks): 시퀀스 처리

<p align="center"><img src="images/img-2026-10-07-09-22-51.png" width="100%"/></p>

예: 이미지 캡셔닝  
이미지 → 단어 시퀀스

<p align="center"><img src="images/img-2026-10-07-09-23-44.png" width="100%"/></p>

예: 감성 분류  
단어 시퀀스 → 감성

<p align="center"><img src="images/img-2026-10-07-09-55-58.png" width="100%"/></p>

예: 기계 번역  
단어 시퀀스 → 단어 시퀀스

<p align="center"><img src="images/img-2026-10-07-09-24-16.png" width="100%"/></p>

예: 프레임 수준의 비디오 분류

### 비시퀀스 데이터의 순차 처리

일련의 "glimpse"를 사용하여 이미지를 분류

<p align="center"><img src="images/img-2026-10-07-09-25-54.png" width="60%"/></p>

이미지를 한 조각씩 생성!

<p align="center"><img src="images/img-2026-10-07-09-26-16.png" width="100%"/></p>

### 순환 신경망

<p align="center"><img src="images/img-2026-10-07-09-27-22.png" width="30%"/></p>

일반적으로 일부 시간 단계에서 벡터를 예측하고자 함

<p align="center"><img src="images/img-2026-10-07-09-27-57.png" width="30%"/></p>

### 순환 신경망 - 펼친 버전

<p align="center"><img src="images/img-2026-10-07-09-29-27.png" width="70%"/></p>

벡터 시퀀스 `x`는 모든 시간 단계에서 순환 공식을 적용하여 처리할 수 있음:

$$
h_t = f_W(h_{t-1}, x_t)
$$

- 새로운 상태
- 이전 상태
- 특정 시간 단계의 입력 벡터
- 매개변수 `W`를 갖는 어떤 함수

<p align="center"><img src="images/img-2026-10-07-09-29-54.png" width="100%"/></p>

벡터 시퀀스 `x`는 모든 시간 단계에서 순환 공식을 적용하여 처리할 수 있음:

$$
h_t = f_W(h_{t-1}, x_t)
$$

주의: 모든 시간 단계에서 동일한 함수와 동일한 매개변수 집합을 사용함.

### (바닐라) 순환 신경망

상태는 하나의 "hidden" 벡터 `h`로 구성됨:

$$
h_t = f_W(h_{t-1}, x_t)
$$

$$
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)
$$

$$
y_t = W_{hy}h_t
$$

<p align="center"><img src="images/img-2026-10-07-09-32-01.png" width="100%"/></p>

### RNN: 계산 그래프

<p align="center"><img src="images/img-2026-10-07-09-32-12.png" width="100%"/></p>

모든 시간 단계에서 동일한 가중치 행렬을 재사용

<p align="center"><img src="images/img-2026-10-07-09-32-26.png" width="100%"/></p>

### RNN: 계산 그래프 - 다대다

<p align="center"><img src="images/img-2026-10-07-09-34-16.png" width="80%"/></p>

### RNN: 계산 그래프 - 다대일

<p align="center"><img src="images/img-2026-10-07-09-33-58.png" width="80%"/></p>

### RNN: 계산 그래프 - 일대다

<p align="center"><img src="images/img-2026-10-07-09-35-05.png" width="80%"/></p>

### Sequence to Sequence: 다대일 + 일대다

다대일: 입력 시퀀스를 하나의 벡터로 인코딩

일대다: 하나의 입력 벡터에서 출력 시퀀스를 생성

문맥 값(context value)이라고 부름 (cf. 해시 값)

<p align="center"><img src="images/img-2026-10-07-09-35-32.png" width="100%"/></p>

### 예: 문자 수준 언어 모델

어휘:

```text
[h, e, l, o]
```

학습 시퀀스 예:

```text
"hello"
```
<p align="center"><img src="images/img-2026-10-07-09-46-44.png" width="70%"/></p>

$$
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)
$$

<p align="center"><img src="images/img-2026-10-07-09-47-03.png" width="70%"/></p>

테스트 시에는 문자를 한 번에 하나씩 샘플링하고 모델에 다시 입력

<p align="center"><img src="images/img-2026-10-07-09-47-47.png" width="60%"/></p>

### 시간 역전파

전체 시퀀스에 대해 순전파하여 손실을 계산한 다음, 전체 시퀀스에 대해 역전파하여 그래디언트를 계산

<p align="center"><img src="images/img-2026-10-07-09-49-43.png" width="100%"/></p>

### 절단 시간 역전파

전체 시퀀스 대신 시퀀스의 청크 단위로 순전파와 역전파를 수행

<p align="center"><img src="images/img-2026-10-07-09-59-09.png" width="50%"/></p>

hidden state는 시간 방향으로 계속 전달하지만, 역전파는 더 적은 수의 단계에 대해서만 수행

<p align="center"><img src="images/img-2026-10-07-09-59-28.png" width="60%"/></p>

<p align="center"><img src="images/img-2026-10-07-09-59-40.png" width="100%"/></p>

### `min-char-rnn.py` gist: Python 112줄

```text
https://gist.github.com/karpathy/d4dee566867f8291f086
```

```python
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
```

<p align="center"><img src="images/img-2026-10-07-10-00-51.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-01-01.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-01-11.png" width="100%"/></p>

### The Stacks Project: 오픈 소스 대수기하학 교재

<p align="center"><img src="images/img-2026-10-07-10-01-57.png" width="100%"/></p>

LaTeX 소스

<p align="center"><img src="images/img-2026-10-07-10-02-07.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-02-23.png" width="100%"/></p>

### 생성된 C 코드

<p align="center"><img src="images/img-2026-10-07-10-02-32.png" width="100%"/></p>

### 해석 가능한 셀 찾기

<p align="center"><img src="images/img-2026-10-07-10-03-04.png" width="100%"/></p>

- 초록색 : hidden state (vector)
  - 그 안의 동그라미 : dimension 하나

Karpathy, Johnson, Fei-Fei: *순환 네트워크 시각화 및 이해*, ICLR Workshop 2016

특정 조건을 만족하면 → 빨간색

<p align="center"><img src="images/img-2026-10-07-10-10-45.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-10-56.png" width="100%"/></p>

인용부호 감지 셀

<p align="center"><img src="images/img-2026-10-07-10-11-07.png" width="100%"/></p>

줄 길이 추적 셀

<p align="center"><img src="images/img-2026-10-07-10-11-18.png" width="100%"/></p>

`if` 문 셀

<p align="center"><img src="images/img-2026-10-07-10-11-32.png" width="100%"/></p>

인용부호/주석 셀

<p align="center"><img src="images/img-2026-10-07-10-11-42.png" width="100%"/></p>

코드 깊이 셀

### 이미지 캡셔닝

<p align="center"><img src="images/img-2026-10-07-10-11-58.png" width="50%"/></p>

- Mao et al., *멀티모달 순환 신경망으로 이미지 설명하기*
- Karpathy and Fei-Fei, *이미지 설명 생성을 위한 심층 시각-의미 정렬*
- Vinyals et al., *Show and Tell: 신경망 이미지 캡션 생성기*
- Donahue et al., *시각 인식 및 설명을 위한 장기 순환 합성곱 네트워크*
- Chen and Zitnick, *이미지 캡션 생성을 위한 순환 시각 표현 학습*

<p align="center"><img src="images/img-2026-10-07-10-12-51.png" width="100%"/></p>

- 초록 : 순환 신경망
- 빨강 : 합성곱 신경망

<p align="center"><img src="images/img-2026-10-07-10-32-24.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-32-56.png" width="100%"/></p>

이전:

$$
h = \tanh(W_{xh} * x + W_{hh} * h)
$$

현재:

$$
h = \tanh(W_{xh} * x + W_{hh} * h + W_{ih} * v)
$$

<p align="center"><img src="images/img-2026-10-07-10-33-14.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-33-25.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-33-37.png" width="100%"/></p>

<p align="center"><img src="images/img-2026-10-07-10-33-45.png" width="100%"/></p>

`<END>` 토큰을 샘플링  
⇒ 종료.

### 이미지 캡셔닝: 결과 예시

<p align="center"><img src="images/img-2026-10-07-10-34-06.png" width="100%"/></p>

- 바닥의 여행 가방 위에 앉아 있는 고양이
- 나뭇가지 위에 앉아 있는 고양이
- 풀밭에서 프리스비를 물고 달리는 개
- 풀밭에 앉아 있는 흰색 테디베어
- 서핑보드를 들고 해변을 걷는 두 사람
- 코트에서 경기 중인 테니스 선수
- 풀밭에 서 있는 두 마리의 기린
- 흙 트랙에서 더트 바이크를 타는 남자

### 이미지 캡셔닝: 실패 사례

<p align="center"><img src="images/img-2026-10-07-10-34-30.png" width="100%"/></p>

- 여성이 손에 고양이를 들고 있음
- 해변에서 서핑보드를 들고 서 있는 여성
- 나뭇가지에 앉아 있는 새
- 책상 위에서 컴퓨터 마우스를 들고 있는 사람
- 야구 유니폼을 입은 남성이 공을 던지고 있음

### 다층 RNN, LSTM

<p align="center"><img src="images/img-2026-10-07-10-34-55.png" width="100%"/></p>

## 바닐라 RNN 그래디언트 흐름

<p align="center"><img src="images/img-2026-10-07-10-35-22.png" width="100%"/></p>

Bengio et al., *그래디언트 하강으로 장기 의존성을 학습하는 것은 어렵다*, IEEE Transactions on Neural Networks, 1994  
Pascanu et al., *순환 신경망 학습의 어려움에 대하여*, ICML 2013

<p align="center"><img src="images/img-2026-10-07-10-35-39.png" width="100%"/></p>

`h_t`에서 `h_{t-1}`로 역전파할 때 `W`를 곱함  
(실제로는 $W_{hh}^{T}$)

<p align="center"><img src="images/img-2026-10-07-10-36-00.png" width="100%"/></p>

`h_0`의 그래디언트를 계산할 때 `W`의 많은 인수와 반복된 `tanh`가 포함됨

가장 큰 특이값 > 1: **그래디언트 폭발**

가장 큰 특이값 < 1: **그래디언트 소실**

그래디언트 클리핑(Gradient Clipping): 그래디언트의 노름이 너무 크면 그래디언트를 스케일링

```text
grad_norm = np.sum(grad * grad)
if grad_norm > threshold:
    grad *= (threshold / grad_norm)
```

→ RNN 구조 변경

### 장단기 메모리 (LSTM)

바닐라 RNN / LSTM

Hochreiter and Schmidhuber, *장단기 메모리*, Neural Computation 1997

[이미지]

## 장단기 메모리 (LSTM)

[Hochreiter et al., 1997]

아래에서 오는 벡터 (`x`)

이전에서 오는 벡터 (`h`)

- `f`: Forget gate, 셀을 지울지 여부
- `i`: Input gate, 셀에 쓸지 여부
- `g`: Gate gate(?), 셀에 얼마나 쓸지
- `o`: Output gate, 셀을 얼마나 드러낼지

- sigmoid
- sigmoid
- sigmoid
- tanh

[이미지]

[이미지]

[이미지]

## 장단기 메모리 (LSTM): 그래디언트 흐름

[Hochreiter et al., 1997]

`c_t`에서 `c_{t-1}`로 역전파할 때는 `f`와의 원소별 곱만 수행하며, `W`와의 행렬 곱은 없음

[이미지]

중단되지 않는 그래디언트 흐름!

[이미지]

# 어텐션

어텐션은 컴퓨터 비전에서 시작됨 (Larochelle & Hinton, 2010)

[Bahdanau et al., *정렬과 번역을 공동으로 학습하는 신경 기계 번역*, ICLR 2015]

## 일반적인 정의

벡터 값들의 집합과 하나의 쿼리 벡터가 주어졌을 때, 어텐션은 쿼리에 따라 값들의 가중합을 계산하는 기법임.

예:

- 디코더 hidden state: query
- 인코더 hidden states: key
- 가중합: value

## 어텐션

### Seq2Seq RNN 모델

$$
p(y_t \mid y_1, y_2, \ldots, y_{t-1}, c)
= g(y_{t-1}, s_t, c)
$$

- `c`: 문맥 값(context value)
- `g`: $y_t$의 확률을 출력하는 비선형 다층 함수
- $s_t$: hidden state
- $y_1, y_2, \ldots, y_{t-1}$: 이전에 예측한 모든 단어

### 어텐션을 사용하는 RNN 모델 - 긴 문장용

$$
p(y_i \mid y_1, y_2, \ldots, y_{i-1}, x)
= g(y_{i-1}, s_i, c_i)
$$

- $c_i$: 문맥 벡터, annotation $(h_1, h_2, \ldots, h_{T_x})$에 의존
- `g`: $y_t$의 확률을 출력하는 비선형 다층 함수
- $s_i$: hidden state
- $x = x_1, x_2, \ldots, x_{T_x}$: 입력 문장

## 어텐션 구현 (Seq2Seq)

[Bahdanau et al., *정렬과 번역을 공동으로 학습하는 신경 기계 번역*, ICLR 2015]

```text
il a m’ entarté
<START> he hit me with a pie
he hit me with a pie <END>
```

- 소스 문장 인코딩
- 소스 문장 (`x`)
  - Word2Vec으로 표현됨
- 타깃 문장 (`y`)
- Encoder RNN
- Decoder RNN

[이미지]

이 부분은 소스 문장에 대한 모든 정보를 담아야 함.

**병목!!!**

- 소스 문장 (`x`)
- 타깃 문장 (`y`)
- Encoder RNN
- Decoder RNN

[이미지]

## 어텐션을 사용하는 RNN 모델

$$
p(y_i \mid y_1, y_2, \ldots, y_{i-1}, x)
= g(y_{i-1}, s_i, c_i)
$$

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

$c_i$: $h_1, h_2, \ldots, h_{T_x}$의 가중합

$$
\alpha_{ij}
= \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$

타깃 단어 $y_i$가 소스 단어 $x_i$와 정렬되어 있을, 즉 그 단어에서 번역되었을 확률.

$$
e_{ij} = a(s_{i-1}, h_j)
$$

- $e_{ij}$: 점수(score)
- $e_{ij}$: 정렬 모델(alignment model) - 위치 `j` 주변의 입력과 위치 `i`의 출력이 얼마나 잘 일치하는지

입력 문장을 벡터 시퀀스로 인코딩하고, 디코딩하는 동안 이 벡터들의 일부를 적응적으로 선택함.

또는, 디코더의 어텐션 메커니즘.

## 어텐션 구현 (어텐션을 사용하는 Seq2Seq)

[Bahdanau et al., *정렬과 번역을 공동으로 학습하는 신경 기계 번역*, ICLR 2015]

```text
il a m’ entarté
<START>
```

- 소스 문장 (`x`)
- Encoder RNN
- Decoder RNN
- 어텐션 점수
- 내적(dot product)

$$
e_{ij} = a(s_{i-1}, h_j)
$$

또는

$$
e_{ij} = a(s_i, h_j)
$$

[이미지]

[이미지]

[이미지]

[이미지]

어텐션 분포

이 디코딩 시간 단계에서는 첫 번째 인코더 hidden state에 대부분 집중하고 있음.

점수에 softmax를 적용하여 확률 분포로 변환함.

$$
\alpha_{ij}
= \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$

[이미지]

어텐션 분포를 사용하여 인코더 hidden state들의 가중합을 계산함.

어텐션 출력은 높은 어텐션을 받은 hidden state의 정보를 주로 포함함.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j
$$

- 어텐션 분포
- 어텐션 출력

[이미지]

어텐션 출력을 디코더 hidden state와 연결한 다음, 이전과 같이 $\hat{y}_1$을 계산하는 데 사용함.

```text
he
```

- 어텐션 분포
- 어텐션 출력

[이미지]

때로는 이전 단계의 어텐션 출력을 사용함.

```text
<START> he
hit
```

- 어텐션 분포
- 어텐션 출력

[이미지]

```text
<START> he hit
me
```

- 어텐션 분포
- 어텐션 출력

[이미지]

```text
<START> he hit me
with
```

그리고 계속 진행

[이미지]

## 어텐션은 훌륭함

어텐션은 신경 기계 번역 성능을 크게 향상시킴.

- 디코더가 소스의 특정 부분에 집중할 수 있게 해 주므로 매우 유용함.

어텐션은 병목 문제를 해결함.

- 어텐션을 사용하면 디코더가 소스를 직접 볼 수 있음.

어텐션은 그래디언트 소실 문제에 도움이 됨.

- 멀리 떨어진 state로 가는 지름길을 제공함.

어텐션은 어느 정도의 해석 가능성을 제공함.

- 어텐션 분포를 살펴보면 디코더가 무엇에 집중했는지 볼 수 있음.
- 정렬(alignment)을 별도의 비용 없이 얻을 수 있음.
- 명시적으로 정렬 시스템을 학습시킨 적이 없다는 점에서 흥미로움.
- 네트워크가 스스로 정렬을 학습함.

어텐션 점수 시각화

[이미지]

## 예: 영어 → 프랑스어 번역

입력:

```text
The agreement on the European Economic Area was signed in August 1992.
```

출력:

```text
L’accord sur la zone economique europeenne a ete signe en aout 1992
```

[이미지]

단어들이 순서대로 대응함.

[이미지]

어텐션이 서로 다른 단어 순서를 파악함.

[이미지]

동사 활용

[이미지]

## 어텐션 구현 (BiRNN)

[Bahdanau et al., *정렬과 번역을 공동으로 학습하는 신경 기계 번역*, ICLR 2015]

BiRNN (Schuster and Paliwal, 1997)은 앞에 있는 단어뿐 아니라 뒤에 있는 단어까지 요약하기 위해 사용됨.

### Forward RNN $\vec{f}$

$\vec{f}$: 입력 문장을 $x_1$에서 $x_{T_x}$까지 읽고, forward hidden states $\vec{h}_1, \vec{h}_2, \ldots, \vec{h}_{T_x}$를 계산함.

### Backward RNN $\overleftarrow{f}$

$\overleftarrow{f}$: 입력 문장을 $x_{T_x}$에서 $x_1$까지 읽고, backward hidden states $\overleftarrow{h}_1, \overleftarrow{h}_2, \ldots, \overleftarrow{h}_{T_x}$를 계산함.

$h_j$: forward와 backward hidden state를 연결함.

$$
h_j = [\vec{h}_j^{T}; \overleftarrow{h}_j^{T}]^{T}
$$

[이미지]

[이미지]

# RNN의 문제점

RNN은 "왼쪽에서 오른쪽"으로 펼쳐짐.

이는 선형적인 지역성을 인코딩함: 유용한 휴리스틱!

가까이 있는 단어들은 서로의 의미에 영향을 주는 경우가 많음.

문제:

RNN에서 멀리 떨어진 단어 쌍이 상호작용하려면 `O(sequence length)` 단계가 필요함.

→ 그래디언트 문제 때문에 장거리 의존성을 학습하기 어려움.

```text
The chef who was
```

`chef`의 정보는 `O(sequence length)`개의 많은 계층을 거쳐 전달됨.

[이미지]

순전파와 역전파에는 `O(sequence length)`개의 병렬화할 수 없는 연산이 있음.

GPU는 서로 독립적인 많은 계산을 동시에 수행할 수 있음.

→ 하지만 과거 RNN hidden state가 계산되기 전에 미래 RNN hidden state를 완전히 계산할 수 없음.

→ 매우 큰 데이터셋에서 학습하는 것을 방해함.

숫자는 state를 계산할 수 있기 전까지 필요한 최소 단계 수를 나타냄.

[이미지]

# 어텐션은 어떨까?

- 어텐션은 각 단어의 표현을 query로 취급하여 값들의 집합에서 정보에 접근하고 이를 통합함.
- 디코더에서 인코더로의 어텐션.
  - 이제 하나의 문장 내부에서 어텐션을 적용.
- 병렬화할 수 없는 연산의 수는 시퀀스 길이에 따라 증가하지 않음.
- 최대 상호작용 거리: `O(1)`. 모든 단어가 모든 계층에서 서로 상호작용하기 때문.

이전 계층의 모든 단어가 모든 단어에 어텐션함. 여기에서는 대부분의 화살표가 생략되어 있음.

```text
I am a boy and …
```

[이미지]

# 셀프 어텐션

- 다시 떠올리기: 어텐션은 query, key, value에 대해 동작함.

$$
\text{queries } q_1, q_2, \ldots, q_T, \quad q_i \in \mathbb{R}^d
$$

$$
\text{keys } k_1, k_2, \ldots, k_T, \quad k_i \in \mathbb{R}^d
$$

$$
\text{values } v_1, v_2, \ldots, v_T, \quad v_i \in \mathbb{R}^d
$$

query의 개수는 key와 value의 개수와 다를 수 있음.

- 셀프 어텐션에서는 query, key, value가 동일한 소스에서 나옴.

예: 이전 계층의 출력이 $x_1, x_2, \ldots, x_T$이고 단어마다 하나의 벡터가 있다면

$$
v_i = k_i = q_i = x_i
$$

- 내적 셀프 어텐션 연산은 다음과 같음.

$$
e_{ij} = q_i^T k_j
$$

key-query 유사도

$$
\alpha_{ij}
= \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$

softmax  
(어텐션 가중치)

$$
output_i = \sum_{j=1}^{T_x}\alpha_{ij}v_j
$$

value들의 가중합

## 셀프 어텐션 블록

- 셀프 어텐션 블록은 LSTM 계층처럼 쌓임.
- 셀프 어텐션이 recurrence를 대체할 수 있을까?
  - 아니오
- 첫 번째 문제: 순서에 대한 내재적인 개념이 없음.

```text
The chef who … food
```

셀프 어텐션은 입력의 순서를 알지 못함.

[이미지]

## 셀프 어텐션의 장벽과 해결책

| 장벽 | 해결책 |
|---|---|
| 순서에 대한 내재적인 개념이 없음 | |

## 시퀀스 순서 해결책

문장의 순서를 key, query, value에 인코딩해야 함.

시퀀스 인덱스를 벡터로 표현:

$$
p_i \in \mathbb{R}^d, \quad i \in \{1,2,\ldots,T_x\}
$$

이 위치 정보를 셀프 어텐션 블록에 어떻게 포함할 것인가?

$p_i$를 입력에 더함. $\tilde{v}_i$, $\tilde{q}_i$, $\tilde{k}_i$는 기존의 value, query, key임.

$$
v_i = \tilde{v}_i + p_i
$$

$$
q_i = \tilde{q}_i + p_i
$$

$$
k_i = \tilde{k}_i + p_i
$$

이것은 첫 번째 계층에서 추가됨.

연결(concatenate)할 수도 있지만, 대부분은 더함.

### 추측 #1: 단순히 세기 - 절대 위치 인코딩

```text
a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇
0  1  2  3  4  5  6  7
```

장점:

- 단순하고 쉬움.

단점:

- 숫자의 규모가 매우 커질 수 있음. (500개 토큰이면 0~499)
- → 벡터 임베딩이 깨질 수 있음.
- → 신경망의 가중치가 크게 변동할 수 있음.

### 추측 #2: 단순히 세고 정규화 - 절대 위치 인코딩

```text
a₀  a₁   a₂   a₃   a₄   a₅   a₆   a₇
0   .14  .29  .43  .57  .71  .86   1
```

장점:

- 단순하고 쉬움.
- 매우 큰 숫자 문제 없음.

단점:

- 동일한 숫자가 시퀀스 길이에 따라 완전히 다른 의미를 가짐.
- → 예: `0.8 = 4/5`는 길이 5인 시퀀스에서는 4번째 원소이지만, 길이 20에서는 16번째 원소임. (`0.8 = 16/20`)
- → 단순한 정규화는 가변 시퀀스 길이에서는 작동하지 않음.

### 추측 #3: 이진 벡터

```text
a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇
0  0  0  0  1  1  1  1
0  0  1  1  0  0  1  1
0  1  0  1  0  1  0  1
```

- $d_{model}$
- 시퀀스 길이

장점:

- 벡터화된 형태
- 범위가 0과 1 사이임.

단점:

- 불연속적인 이산 값

불연속적인 이산 값 → 연속적인 무언가가 필요함.

## 시퀀스 순서 해결책 - 사인파를 통한 벡터

사인파 위치 표현: 주기가 서로 다른 사인 함수를 연결함.

$$
p_i =
\begin{bmatrix}
\sin(i / 10000^{2\times1/d}) \\
\cos(i / 10000^{2\times1/d}) \\
\vdots \\
\sin(i / 10000^{2\times(d/2)/d}) \\
\cos(i / 10000^{2\times(d/2)/d})
\end{bmatrix}
$$

- 차원
- 시퀀스 내 인덱스
- 입력 및 출력 크기

장점:

- 주기성은 아마도 "절대 위치"가 그다지 중요하지 않을 수 있음을 나타냄.
- 주기가 다시 시작되므로 더 긴 시퀀스로 외삽할 수도 있음.

단점:

- 학습 가능하지 않음.
- 또한 외삽은 실제로 잘 작동하지 않음.

```text
I        p₀
am       p₁
a        p₂
student  p₃
```

입력 + 순서

[이미지]

$$
p_i =
\begin{bmatrix}
\sin(i / 10000^{2\times1/d}) \\
\cos(i / 10000^{2\times1/d}) \\
\vdots \\
\sin(i / 10000^{2\times(d/2)/d}) \\
\cos(i / 10000^{2\times(d/2)/d})
\end{bmatrix}
$$

[이미지]

### 시퀀스 순서 해결책 - 예시

$$
p_i =
\begin{bmatrix}
\sin(i / 100^{2\times0/d}) \\
\cos(i / 100^{2\times0/d}) \\
\vdots \\
\sin(i / 100^{2\times(d/2-1)/d}) \\
\cos(i / 100^{2\times(d/2-1)/d})
\end{bmatrix}
\rightarrow
\begin{bmatrix}
\sin(i/1) \\
\cos(i/1) \\
\vdots \\
\sin(i/10) \\
\cos(i/10)
\end{bmatrix}
$$

[이미지]

## 학습된 절대 위치 표현

행렬 $p \in \mathbb{R}^{d\times T}$를 학습하고, 각 $p_i$를 해당 행렬의 한 열로 사용함.

대부분의 시스템이 이것을 사용함!

시퀀스 인덱스를 벡터로 표현:

$$
p_i \in \mathbb{R}^d, \quad i \in \{1,2,\ldots,T_x\}
$$

장점:

- 유연성: 각 위치가 데이터에 맞도록 학습될 수 있음.

단점:

- `1, …, T` 범위를 벗어난 인덱스로는 확실히 외삽할 수 없음.

더 유연한 위치 표현:

- 상대적 선형 위치 어텐션 [Shaw et al., 2018, `https://arxiv.org/abs/1803.021155`]
- 의존 구문 기반 위치 [Wang et al., 2019, `https://arxiv.org/pdf/1909.00383.pdf`]

## 셀프 어텐션의 장벽과 해결책

| 장벽 | 해결책 |
|---|---|
| 순서에 대한 내재적인 개념이 없음 | 입력에 위치 정보를 추가 |
| 딥러닝을 위한 비선형성이 없음 (단순한 가중 평균…) | |

## 셀프 어텐션에 비선형성 추가

- 셀프 어텐션에는 원소별 비선형성이 없음.
- 셀프 어텐션 계층을 쌓는 것만으로는 value 벡터를 다시 평균낼 뿐임.
- 간단한 해결책:
  - 각 출력 벡터를 후처리하기 위해 feed-forward network를 추가.

$$
m_i = MLP(output_i)
$$

$$
= W_2 \times ReLU(W_1 \times output_i + b_i) + b_2
$$

FF 네트워크가 어텐션 결과를 처리함.

[이미지]

## 셀프 어텐션의 장벽과 해결책

| 장벽 | 해결책 |
|---|---|
| 순서에 대한 내재적인 개념이 없음 | 입력에 위치 정보를 추가 |
| 딥러닝을 위한 비선형성이 없음 (단순한 가중 평균…) | 각 셀프 어텐션 출력에 동일한 FF 네트워크를 적용 |
| "미래를 볼 수 없음" | |

## 셀프 어텐션에서 미래 마스킹

- 디코더에서 셀프 어텐션을 사용하려면 미래를 보는 것이 금지됨.
- 모든 시간 단계에서 key와 query의 집합을 변경하여 과거 단어만 포함하도록 함.
- 간단한 해결책:
  - 어텐션 점수를 $-\infty$로 설정하여 어텐션을 마스킹함.

$$
e_{ij} =
\begin{cases}
q_i^T k_j, & j < i \\
-\infty, & j \ge i
\end{cases}
$$

이 단어들을 인코딩하기 위해

```text
[START] The chef who
```

볼 수 있는 단어들(회색 처리되지 않은 단어들)

$e_{ij}$ 값의 행렬

[이미지]

이 단어들을 디코딩하기 위해

```text
[START]
The
chef
who
```

[이미지]

## 셀프 어텐션의 장벽과 해결책

| 장벽 | 해결책 |
|---|---|
| 순서에 대한 내재적인 개념이 없음 | 입력에 위치 정보를 추가 |
| 딥러닝을 위한 비선형성이 없음 (단순한 가중 평균…) | 각 셀프 어텐션 출력에 동일한 FF 네트워크를 적용 |
| "미래를 볼 수 없음" | 어텐션 가중치를 인위적으로 0으로 설정하여 미래를 마스킹 |

## 셀프 어텐션 블록에 필요한 요소

| 장벽 | 해결책 |
|---|---|
| 순서에 대한 내재적인 개념이 없음 | 입력에 위치 정보를 추가 |
| 딥러닝을 위한 비선형성이 없음 (단순한 가중 평균…) | 각 셀프 어텐션 출력에 동일한 FF 네트워크를 적용 |
| "미래를 볼 수 없음" | 어텐션 가중치를 인위적으로 0으로 설정하여 미래를 마스킹 |

# Transformer Encoder-Decoder

[*Attention Is All You Need*, Vaswani et al., 2017, `https://arxiv.org/pdf/1706.03762.pdf`]

## 높은 수준에서 본 Transformer

- 입력 시퀀스
- Encoder
- Encoder
- 단어 임베딩
- 위치 정보
- `+`
- Decoder
- Decoder
- 단어 임베딩
- 위치 정보
- `+`
- 출력 시퀀스
- Decoder가 encoder state에 어텐션함
- 예측!

[이미지]

## 현재 남아 있는 문제

1. K, Q, V 어텐션: 하나의 단어 임베딩에서 `k`, `q`, `v` 벡터 얻기.
2. Multi-headed attention: 하나의 계층에서 여러 위치에 어텐션하기.
3. 학습 기법
   - Residual connection
   - Layer normalization
   - 내적 스케일링

# Transformer: Key-Query-Value 어텐션

[*Attention Is All You Need*, Vaswani et al., 2017, `https://arxiv.org/pdf/1706.03762.pdf`]

다시 떠올리기: 셀프 어텐션에서는 K, Q, V가 동일한 소스에서 나옴.

어떻게?

$x_1, x_2, \ldots, x_T$를 입력 벡터라고 하자. $x_i \in \mathbb{R}^d$.

$$
k_i = Kx_i, \quad K \in \mathbb{R}^{d\times d}
$$

`K`는 key 행렬.

$$
q_i = Qx_i, \quad Q \in \mathbb{R}^{d\times d}
$$

`Q`는 query 행렬.

$$
v_i = Vx_i, \quad V \in \mathbb{R}^{d\times d}
$$

`V`는 value 행렬.

의미: `x` 벡터의 서로 다른 측면이 세 역할 각각에서 사용되거나 강조됨.

[이미지]

## 행렬을 사용한 계산

입력 벡터를 연결한 것을 다음과 같이 둠.

$$
X = [x_1; \ldots; x_T] \in \mathbb{R}^{T\times d}
$$

$$
XK \in \mathbb{R}^{T\times d}, \quad
XQ \in \mathbb{R}^{T\times d}, \quad
XV \in \mathbb{R}^{T\times d}
$$

query-key 내적:

$$
XQ(XK)^T \in \mathbb{R}^{T\times T}
$$

출력(어텐션 값):

$$
softmax(XQ(XK)^T) \times XV \in \mathbb{R}^{T\times d}
$$

[이미지]

# Transformer: Multi-headed Attention

한 번에 문장의 여러 위치를 보려면?

단어 `i`에 대해, 셀프 어텐션은 다음 값이 높은 위치를 봄.

$$
x_i^TQ^TKx_j
$$

서로 다른 이유로 서로 다른 `j`에 집중하고 싶다면?

여러 개의 Q, K, V 행렬을 사용하여 여러 어텐션 "head"를 만듦.

$$
Q_l, K_l, V_l \in \mathbb{R}^{d\times\frac{d}{h}}
$$

- `h`: 어텐션 head의 개수
- `l`: 1부터 `h`까지
- 계산 부하는 동일함.

- single-head attention (query 행렬만)
- multi-head attention (query 행렬만, 두 개의 head)

[이미지]

각 어텐션 head는 독립적으로 어텐션을 수행함.

$$
output_l = softmax(XQ_l(XK_l)^T) \times XV_l \in \mathbb{R}^{d/h}
$$

그런 다음 모든 head의 출력을 결합함.

$$
output = Y[output_1; \ldots; output_h], \quad
Y \in \mathbb{R}^{d\times\frac{d}{h}}
$$

- `h`: 어텐션 head의 개수
- `l`: 1부터 `h`까지
- 계산 부하는 동일함.

[이미지]

# Transformer: Residual Connection

[*이미지 인식을 위한 심층 잔차 학습*, He et al., 2016, `https://arxiv.org/abs/1512.03385`]

Residual connection은 모델이 더 잘 학습되도록 돕는 기법임.

다음 대신:

$$
X^{(i)} = layer(X^{(i-1)})
$$

여기서 `i`는 계층을 나타냄.

다음을 사용:

$$
X^{(i)} = X^{(i-1)} + layer(X^{(i-1)})
$$

여기서 `i`는 계층을 나타냄.

Residual connection은 loss landscape를 상당히 더 매끄럽게 만드는 것으로 여겨짐.

ResNet-56의 loss surface [*신경망의 Loss Landscape 시각화*, Li et al., 2018]

[이미지]

# [Tip] Residual Learning 상세

[*이미지 인식을 위한 심층 잔차 학습*, He et al., 2016, `https://arxiv.org/abs/1512.03385`]

Residual connection을 사용하는 매우 깊은 네트워크

- ImageNet을 위한 152계층 모델.
- ILSVRC'15 분류 우승 (top-5 error 3.57%)
- ILSVRC'15 및 COCO'15의 모든 분류 및 검출 대회를 석권

## 일반 CNN에 더 깊은 계층을 계속 쌓으면 어떻게 되는가?

(BN을 사용하고 RL은 사용하지 않음)

**과소적합!!**

질문: 이 학습 곡선과 테스트 곡선에서 이상한 점은 무엇인가?

[이미지]

그래디언트 소실 / 폭발 문제?

→ 정규화된 초기화와 중간 정규화 해결책이 있음.

56계층 모델이 학습 오차와 테스트 오차 모두에서 20계층 모델보다 성능이 나쁨.

(degradation problem)

→ 처음에는 깊은 모델이 다른 모델보다 훨씬 크기 때문에 과적합한 것인가?

→ 하지만 과적합 때문에 발생한 것이 아님! 왜?

→ 학습 오차가 너무 높음…

→ 가설: 잘못된 최적화 때문에 발생함. (더 깊은 모델은 최적화하기 더 어려움)

## 가설 검증

상식? 더 깊은 모델은 적어도 더 얕은 모델만큼은 잘 작동할 수 있어야 함.

가설을 검증하기 위해,

더 얕은 모델을 모방하여 비교 가능한 더 깊은 모델을 구성함.

1. 더 얕은 모델에서 학습된 계층을 복사하고
2. 추가 계층을 identity mapping으로 설정함.

[이미지]

1. 더 얕은 모델에서 학습된 계층을 복사하고
2. 추가 계층을 identity mapping으로 설정함.

[이미지]

## 결과는?

더 깊은 모델은 최적화하기 더 어려움. 따라서 얕은 모델을 모방하기 위한 identity function을 학습하기 어려움.

[이미지]

## 해결책은?

원하는 기본 매핑을 직접 맞추려고 하는 대신 네트워크 계층을 사용하여 residual mapping을 찾음.

- Plain block
- Residual block
- Additive shortcut

[이미지]

이 부분들을 0으로 설정하면 전체 블록이 identity function을 계산함!

[이미지]

# Transformer: 전체 구조

[이미지]
