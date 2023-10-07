이 문서는 YOLOX의 INT8, FP16의 Quantization에 관한 문서입니다.

다음과 같은 작업이 필요합니다.

# 환경 세팅

1. torch2trt 환경 세팅이 필요합니다.

torch2trt는 torch파일을 fp16, int8로 변환해 주는 라이브러리로 환경 설정이 까다로워 docker를 추천합니다.

torch2trt/docker/21-06/

폴더에 가서 build.sh를 이용해 torch2trt를 사용할 수 있는 환경을 만듭니다.

2. YOLOX 환경을 마련합니다.

/opt/project 폴더로 이동한뒤 YOLOX를 구동하기 위한 setup.py develop을 실행해 환경을 세팅합니다.

3. int8 calibration을 위한 선작업을 합니다.

fp16은 추가 데이터셋이 필요 없으나 int8은 calibration에 필요한 데이터 셋이 필요합니다. 

tool/trt.py를 확인하면 path를 확인 할수 있는데 사용자 마음대로 사진이 담긴 폴더를 생성해 calibration을 수행하면 됩니다. 전 100장을 사용했습니다.

4. make_trt.sh를 실행합니다.

YOLOX_outputs/ 폴더에 fp16.trt, int8.trt 파일이 생성된것을 확인 할 수 있습니다.

5. cpu gpu fp16 int8의 inference time을 비교합니다.

python run_exam.py를 실행해 result.txt파일을 확인해 비교하게 됩니다.

6. 결과

cpu

Infer time: 0.0347s

Infer time: 0.0368s

Infer time: 0.0357s
gpu

Infer time: 0.9126s

Infer time: 0.8987s

Infer time: 0.9130s

fp 16

Infer time: 0.0025s

Infer time: 0.0027s

Infer time: 0.0025s

int 8

Infer time: 0.0046s

Infer time: 0.0042s

Infer time: 0.0043s

tip:

https://nvidia-ai-iot.github.io/torch2trt/v0.2.0/usage/reduced_precision.html

이 링크를 확인하시면 int8 calibration의 여러 알고리즘 및 방법들을 더 확인해 볼수 있습니다. tool/trt.py를 수정해 다양한 작업이 가능합니다.