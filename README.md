<p align="center">
  <img src="https://github.com/user-attachments/assets/9b5a9ad5-01d1-4227-94cb-93515b99b4b3" height="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/90f4cc03-f857-4bba-bfa2-a24404f39be3" height="25">
</p>  

*본 연구의 성과물은 연구 개발, 교육 등 비영리 목적으로 활용되는 경우 공개 라이선스를 통해 공개하지만, 사업화, 특허 등 영리 목적 활용의 경우 상용화 라이선스를 적용함.*   
   
   
# Personality AI, 'PAI'

PAI(Personality AI)는 '개성 형성이 가능한 에이전트 플랫폼 기술 개발' 연구를 위한 프로젝트로, 인공지능에 인간의 개성을 부여하는 연구를 목표로 하고 있습니다.   

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
