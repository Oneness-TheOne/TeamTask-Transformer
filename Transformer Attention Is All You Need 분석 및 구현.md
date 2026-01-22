# 🚀 Transformer: Attention Is All You Need 분석 및 구현

본 프로젝트는 2017년 Google에서 제안한 혁신적인 신경망 아키텍처인 **Transformer**를 PyTorch로 밑바닥부터 직접 구현하고, **AG News** 데이터셋을 활용하여 뉴스 카테고리 분류 성능을 검증한 팀 과제입니다.

---

## 1. 논문 전체 구조 리뷰 (이론 분석)

### 1.1 RNN/LSTM과의 구조적 차이점: 왜 트랜스포머인가?

- **RNN의 한계**: 데이터를 순차적으로 처리해야 하는 **순차적 제약**이 있습니다. 문장이 길어질수록 앞쪽 정보가 희석되는 **Gradient Vanishing** 문제가 발생하며, 이전 시점의 계산이 끝나야 다음 시점을 계산할 수 있어 병렬 처리가 불가능합니다.
- **Transformer의 혁신**: **순환 구조를 완전히 제거**했습니다. 문장 전체를 한 번에 입력받아 병렬로 처리하므로 학습 속도가 비약적으로 빠르며, 거리에 상관없이 단어 간의 전역적 의존성(Global Dependency)을 직접 계산합니다.

### 1.2 핵심 모듈 분석: 왜 이런 구조가 필요한가?

| **모듈 명칭** | **필요성 (Why?)** | **작동 원리 (How?)** |
| --- | --- | --- |
| **Self-Attention** | 단어의 의미는 주변 단어와의 관계에 의해 결정됨 | 문장 내 모든 단어 쌍 간의 유사도를 점수화하여 관련 높은 정보에 집중 |
| **Multi-Head Attention** | 문법, 의미 등 다양한 관점의 문맥 파악이 필요함 | 어텐션을 여러 개(Head)로 나누어 병렬 수행하여 다각도의 특징 추출 |
| **Positional Encoding** | 병렬 처리 방식은 순서 정보가 없어 위치 정보 주입이 필수 | 사인/코사인 함수 기반의 위치 값을 더해 단어의 상대적/절대적 위치 인식 |
| **Feed Forward (FFN)** | 어텐션 결과에 비선형성을 추가하고 특징을 변환해야 함 | 각 단어 벡터에 독립적으로 적용되는 두 층의 선형 변환과 활성화 함수 |
| **Residual & Layer Norm** | 층이 깊어질 때 발생하는 학습 불안정과 정보 소실 방지 | 이전 층의 정보를 직접 전달(잔차)하고 데이터를 정규화하여 학습 가속화 |

---

## 2. 구현 상세 (Implementation Details)

`nn.Transformer` API를 사용하지 않고 다음과 같이 클래스 단위로 직접 구현하였습니다.

- **모델 구조**: `NewsClassifier` (Encoder 기반 모델)
- **핵심 구현 클래스**:
    - `PositionalEncoding`: 시퀀스 내 상대적 위치 정보 주입
    - `MultiHeadAttention`: 쿼리, 키, 값을 여러 헤드로 나누어 병렬 관계 추출
    - `EncoderLayer`: Self-Attention과 FFN의 유기적 결합 블록
- **특이사항**: 패딩 토큰(`[PAD]`)이 어텐션 연산에 포함되지 않도록 **Padding Mask**를 구현하였으며, 분류 성능을 높이기 위해 **Global Average Pooling**을 적용했습니다.

---

## 3. 실험 결과 요약

### 3.1 실험 설정 및 로그

- **데이터셋**: AG News (World, Sports, Business, Sci/Tech 4종 분류)
- **학습 설정**: 10,000개 샘플 활용, 10 Epochs 학습

```jsx
학습 시작 (Device: cuda)...
Epoch 1/10 | Loss: 1.3184
Epoch 2/10 | Loss: 1.0021
Epoch 3/10 | Loss: 0.7693
Epoch 4/10 | Loss: 0.6483
Epoch 5/10 | Loss: 0.5799
Epoch 6/10 | Loss: 0.5223
Epoch 7/10 | Loss: 0.4820
Epoch 8/10 | Loss: 0.4543
Epoch 9/10 | Loss: 0.4221
Epoch 10/10 | Loss: 0.4019
```

### 3.2 최종 추론 검증

```jsx
--------------------------------------------------
입력 뉴스: The final match of the world cup was incredibly intense.
분류 결과: Sports (정답: Sports)

입력 뉴스: The stock market saw a significant drop after the federal report.
분류 결과: Business (정답: Business)
```

---

## 4. 트랜스포머 확장 및 응용 분석

### 4.1 트랜스포머를 응용한 대표적인 구조

트랜스포머는 각 구성 요소(Encoder, Decoder)의 활용 방식에 따라 다양하게 진화했습니다.

1. **BERT (Bidirectional Encoder Representations from Transformers)**
    - **구조**: 트랜스포머의 **인코더**를 층층이 쌓은 구조.
    - **특징**: 양방향 문맥 정보를 동시에 파악하여 문장 분류, 개체명 인식 등 '이해' 작업에 특화됨.
2. **GPT (Generative Pre-trained Transformer)**
    - **구조**: 트랜스포머의 **디코더**를 활용한 구조.
    - **특징**: 이전 단어들을 바탕으로 다음 단어를 예측하는 Auto-regressive 방식이며, 자연스러운 텍스트 생성에 최적화됨.
3. **ViT (Vision Transformer)**
    - **구조**: 이미지 패치를 시퀀스로 간주하여 트랜스포머 **인코더**에 입력.
    - **특징**: CNN의 고유한 편향(Inductive Bias) 없이도 대규모 데이터에서 이미지 인식 분야의 SOTA를 달성.
4. **T5 (Text-to-Text Transfer Transformer)**
    - **구조**: 트랜스포머의 **인코더-디코더** 구조를 모두 유지.
    - **특징**: 모든 자연어 처리 문제를 "텍스트 입력 -> 텍스트 출력" 형태(예: 번역, 요약, 문답)로 통일하여 해결하는 범용 프레임워크 제시.

### 4.2 트랜스포머가 범용 아키텍처가 된 이유

- **확장성 (Scalability)**: 모델의 크기와 데이터 양을 늘릴수록 성능이 지속적으로 향상되는 'Scaling Law'가 입증되었습니다.
- **유연한 입력 처리**: 텍스트뿐만 아니라 이미지(ViT), 오디오(Whisper), 시계열 데이터 등 '시퀀스'로 표현 가능한 모든 도메인에 적용이 가능합니다.
- **전이 학습 (Transfer Learning)**: 방대한 데이터로 사전 학습(Pre-train)된 모델을 특정 작업에 미세 조정(Fine-tuning)하여 적은 데이터로도 압도적인 성능을 낼 수 있습니다.

### 4.3 트랜스포머의 진화 과정

- **탄생 (2017)**: Seq2Seq 모델의 성능 한계를 극복하기 위해 어텐션 메커니즘만으로 구성된 모델 제안.
- **NLP 정복 (2018~2019)**: BERT와 GPT의 등장으로 자연어 이해 및 생성 분야의 표준으로 자리 잡음.
- **모달리티 확장 (2020~현재)**: Vision Transformer를 기점으로 멀티모달(Multi-modal) 학습의 핵심 엔진으로 성장하며, 현재 거대 언어 모델(LLM) 시대의 근간이 됨.