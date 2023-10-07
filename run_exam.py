import os

# 개사진 실험
# cpu

os.system("echo cpu")

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device cpu')

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device cpu')

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device cpu')

# gpu

os.system("echo gpu")

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device gpu')

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device gpu')

os.system('python tools/demo.py image -n yolox-nano -c \
    ./yolox_nano.pth --save_result --device gpu')

# fp 16

os.system("echo fp 16")

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/fp16.pth')

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/fp16.pth')

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/fp16.pth') 


# int 8.

os.system("echo int 8")

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/int8.pth')

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/int8.pth')

os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
    -c ./YOLOX_outputs/yolox_nano/int8.pth')

# # mp4 

# # cpu

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device cpu --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device cpu --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device cpu --path ./322f79bea1e8d.mp4')

# # gpu

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device gpu --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device gpu --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano -c \
#     ./yolox_nano.pth --save_result --device gpu --path ./322f79bea1e8d.mp4')

# # fp 16

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/fp16.pth --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/fp16.pth --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/fp16.pth --path ./322f79bea1e8d.mp4') 


# # int 8

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/int8.pth --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/int8.pth --path ./322f79bea1e8d.mp4')

# os.system('python tools/demo.py image -n yolox-nano --trt --save_result \
#     -c ./YOLOX_outputs/yolox_nano/int8.pth --path ./322f79bea1e8d.mp4')
