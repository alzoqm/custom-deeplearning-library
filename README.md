# cuda_study
## cuda 코드를 활용한 현재 구현 내용
기본적인 행렬 연산 구현 <br>

23-02<br>
sum, max, min, reduce 코드 구현, Tensor class에 차원 축소 및 확장 기능 추가(squeeze, unsqueeze)<br>
다른 차원의 행렬 연산 지원, Tensor class에 reshape기능 추가<br>
Linear.h 추가<br><br>
23-03<br>
Linear class 생성 및 forward 및 backward 구현 <br>
transpose 구현<br><br>

다음에 추가할 것<br>
matCal 브로그캐스팅 지원
half class 연산 추가 지원<br>
forward와 backward 이후 함수에서 main으로 돌아올 때 많은 시간이 걸리는 것 고치기<br>

23-02 repo 이름 변경<br>
repo rename cuda_study -> deeplearning library <br>
## test with colab
[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" />](https://colab.research.google.com/drive/13DRdZlK3QTPUS_Xy3xhGnW5yLXe_qCwg)
