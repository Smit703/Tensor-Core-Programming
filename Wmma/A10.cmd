nvcc -arch=sm_86 Wmma_gemm_fp16.cu -o wmma_fp16_4096
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Wmma_A10G_fp16_4096 ./wmma_fp16_4096

nvcc -arch=sm_86 Wmma_gemm_int8.cu -o wmma_int8_4096
sudo /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Wmma_A10G_int8_4096 ./wmma_int8_4096
