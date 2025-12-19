<p align="center">
  <img src="https://github.com/user-attachments/assets/9b5a9ad5-01d1-4227-94cb-93515b99b4b3" height="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/90f4cc03-f857-4bba-bfa2-a24404f39be3" height="25">
</p>  

*본 연구의 성과물은 연구 개발, 교육 등 비영리 목적으로 활용되는 경우 공개 라이선스를 통해 공개하지만, 사업화, 특허 등 영리 목적 활용의 경우 상용화 라이선스를 적용함.*       
   
   
# Personality AI, 'PAI'

PAI(Personality AI)는 '개성 형성이 가능한 에이전트 플랫폼 기술 개발' 연구를 위한 프로젝트로,   
인공지능에 인간의 개성을 부여하는 연구를 목표로 하고 있습니다.   

본 연구는 과학기술정보통신부와 정보통신기획평가원의 ‘사람 중심 인공지능 핵심 원천기술 개발’ 사업의 지원을 받아 수행되고 있습니다.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/92108647-1b42-4fc1-8621-d77090d3ee88" height="450">
</p>

<br/>

## 관련 기사  
- '한·프·일, 교감형 AI 공동연구…AI에 인간 개성 부여', 이데일리  
  https://zdnet.co.kr/view/?no=20230920130357  
  
- '영화 ‘her’처럼…AI에 개성을, 100억 규모 국제 공동연구 막올라', 지디넷코리아  
  https://m.edaily.co.kr/News/Read?newsId=02345206635742088&mediaCodeNo=257

- '경북교육청 미래교육 행사에서 AI 교육 솔루션으로서 휴먼 키오스크 공개', E동아
  https://edu.donga.com/news/articleView.html?idxno=100483

<br/>

## 단계별 연구 성과

**1단계 연구 성과**

- [**Big-Five Personality Recognition Model**] (본 레포지토리)  
  자체 구축한 멀티모달 데이터셋으로 학습한 모델을 통해 Big-Five trait로 라벨링된 개성을 분류 및 회귀 방법으로 인식하기 위한 연구

<br/>

- [**Jonathan A-LLM**](https://huggingface.co/AcrylaLLM/Llama-3-8B-Jonathan-aLLM-Instruct-v1.0)  
  **개성 모델의 언어적 정보 기반 연동 서비스를 위해 Meta의 LLaMA-3-8B 기반으로 DoRA를 활용해 미세 조정된 LLM으로, 한국어 LLM 리더보드 오픈소스 1위 달성 (2024년 10월 기준)**   
  Jonathan A-LLM은 Meta의 Llama-3-8B 아키텍처를 기반으로 구축된 한국어 대규모 언어 모델로 포괄적인 한국어 데이터 세트에서 DoRA(Weight-Decomposed Low-Rank Adaptation) 방법론을 사용하여 훈련되었으며, 한국어 이해 및 생성에 최적화되어 Horangi Korean LLM Leaderboard에서 오픈 소스 한국어 모델 중 최첨단 성능을 달성  
  <img src="https://github.com/user-attachments/assets/cf09ce28-f223-418f-9eb8-05c53cb2ac11" height="500">

<br/>
  
- [**Jonathan Flightbase**](https://github.com/AcrylAI/Jonathan-Flightbase)  
  **개성 모델 개발을 위한 데이터 전처리, 학습, 배포 맟 조합까지 하나의 플랫폼에서 수행 가능한 전 주기 지원 AI DevOps 플랫폼**   
  Jonathan은 인공 지능 모델의 학습, 배포를 포함하여 전처리 및 후가공까지의 AI 개발의 End-to-End를 지원하는 AI DevOps 플랫폼으로,  
  개성 데이터 전처리와 개성 모델 조합을 위한 엔진을 추가로 개발하여 다른 모델의 개발 및 테스트를 포함한 배포를 위하여 플랫폼 활용  
  <img src="https://github.com/user-attachments/assets/1b3cc7e0-b80b-4603-bf0a-6d77f7f60e36" height="100">

<br/>

- [**Grounding Visual Representation with Texts(GVRT)**](https://github.com/seonwoo-min/GVRT)  
  멀티모달 임베딩 공간의 도메인 변화에 대한 강건성을 확보하기 위한 연구    
  <img src="https://github.com/user-attachments/assets/d26f1fed-f829-4119-82ef-e26ba64c5c2b" height="200">
  
<br/>
  
- [**Class Aware Text(CAT) generator**](https://github.com/NaaeKwon/CAT)  
  텍스트와 목표 스타일 정보를 입력받아 다양한 텍스트를 생성함으로써 모델 내에서 다차원 개성 정보의 임베딩이 수행될 수 있음을 보여주며, 임베딩 시 잠재 변수의 영향력을 평가하기 위한 연구    
  <img src="https://github.com/user-attachments/assets/7552614a-cd48-4a83-975f-0f56306fa59d" height="200">  

<br/>
  
- [**Stable Baslines with JAX & Haiku**](https://github.com/kwk2696/sb3-jax-haiku)  
  Jax & Haiku 기반의 강화학습/모방학습 알고리즘 라이브러리로, 기존 Torch 기반의 알고리즘에 비해 2배 이상의 학습 속도 향상  
  <img src="https://github.com/user-attachments/assets/bab217d1-9fec-4812-8638-3fc296a4edea" height="250">


<br/>
     
- [**Overview of Personality Trait Prediction**](https://github.com/ISSR-CBNU/Personality-trait-prediction)    
  개성 인식을 위한 CNN 기반의 VGG16, ResNet, Inception V2 모델과, Transformer 기반의 Video Swin Transformer, FAt Transformer, MAE(Masked AutoEncoder), ViVit(Video Vision Transforemr) 모델들의 성능 비교를 위한 연구

<br/>
  
- [**OCEAN Domain Adaptation Language Model**](https://github.com/bytecell/PersonalityAI)    
  개성 인식을 위한 데이터 전처리, 시각화, 언어모델의 사전 학습, 파인 튜닝, 디코더 모델(SOLAR)의 추론을 포함하는 연구

<br/>

**2단계 연구 성과**

- [**Personality Recognition & Regression Pre-trained Model**] (본 레포지토리)  
  과제 전체 결과물 홍보 및 한국인을 대상으로 구축한 데이터로 학습된 개성 인식 분류 및 회귀 사전 학습 모델을 공개하며, 한국인의 self-report 개성 정보가 레이블링된 대규모 데이터셋 구축 결과를 바탕으로 몰입도 높은 개성을 발현하기 위한 정보 및 에이전트의 자기 개선 알고리즘에 활용하기 위한 연구

<br/>

- [**SemanticControl**](https://github.com/mung3477/SemanticControl)  
  사용자의 의도에 따라 이미지를 변형하여 생성할 수 있는 모델로, 개성, 감성 등을 에이전트가 이미지의 형태로 표현하기 위해서 특정 캐릭터의 제스처, 동작, 표정 등을 텍스트 입력으로 자유롭게 변형할 수 있는 모델에 활용하기 위한 연구

<br/>

- [**Tri-layer Contrative Decoding (TCD)**](https://github.com/KR-0822/TCD)  
  Vision-Language Model (VLM)의 환각을 저감하여 답변할 수 있는 모델에 대한 소스 코드로, 멀티모달 LLM을 기반으로 환각현상을 저감하여 사용자와 대화를 나눌 수 있는 시스템에 활용하기 위한 연구

<br/>

- [**Prototype-based Regularization**](https://github.com/psb485/HNPR-FSCIL)  
  멀티모달 임베딩 공간 최적화를 위한 Prototype 기반의 regularization 기법에 대한 소스 코드로, 단일한 임베딩 공간을 활용하여 연속적으로 사용자와 일관성있게 대화를 나누기 위한 임베딩 공간 학습 기법에 활용하기 위한 연구

<br/>

- [**Stream and Query-guided Feature Aggregation**](https://github.com/moonseokha/DuOcc)  
  사용자와의 대화 중에 시각적인 정보를 시간이 지남에 따라 잃어버리지 않고 유지하여 일관성 있는 챗봇을 구현하기 위해, Stream and Query-guided Feature Aggregation 기반의 멀티모달 임베딩 공간 최적화 기법을 제안하는 연구

<br/>

- [**Personality-recognition**](https://github.com/ISSR-CBNU/Personality-recognition)  
  영상 및 음성의 멀티모달 융합 기술과 Transformer 계열 모델을 활용하여 Big-Five(OCEAN) 성격 특성을 예측하고 실시간 인식 소프트웨어에 적용 가능한 딥러닝 프레임워크 ViViT, TimeSformer, AST 등 다양한 백본을 통해 언어적·비언어적 요소를 통합 분석하며, 새로운 모델 제안을 위한 표준 비교 실험(Baseline) 환경 구축 및 성격 인식 모델의 성능 최적화를 수행하는 연구

<br/>

- [**PersonalityAI**](https://github.com/bytecell/PersonalityAI)  
  도메인 적응형 사전학습(DAPT) 기법을 적용하여 텍스트 및 멀티모달 데이터를 기반으로 Big-Five(OCEAN) 성격 특성을 정밀하게 분석하는 개성 인식 언어 모델 및 학습 시스템 인코더·디코더 기반 모델의 도메인 특화 학습과 Instruction Tuning, 멀티모달 융합 튜닝을 지원하며 데이터 전처리부터 시각화 및 추론에 이르는 전 과정을 통합하여 개성 인식 성능을 최적화하는 연구