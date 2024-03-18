nvcc -arch=sm_86 Cublas_gemm_256.cu -o cublas_256 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_256 ./cublas_256

nvcc -arch=sm_86 Cublas_gemm_512.cu -o cublas_512 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_512 ./cublas_512

nvcc -arch=sm_86 Cublas_gemm_1024.cu -o cublas_1024 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_1024 ./cublas_1024

nvcc -arch=sm_86 Cublas_gemm_2048.cu -o cublas_2048 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_2048 ./cublas_2048

nvcc -arch=sm_86 Cublas_gemm_4096.cu -o cublas_4096 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_4096 ./cublas_4096

nvcc -arch=sm_86 Cublas_gemm_1000.cu -o cublas_1000 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_1000 ./cublas_1000

nvcc -arch=sm_86 Cublas_gemm_4000.cu -o cublas_4000 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_4000 ./cublas_4000

nvcc -arch=sm_86 Cublas_gemm_2047.cu -o cublas_2047 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_2047 ./cublas_2047

nvcc -arch=sm_86 Cublas_gemm_2049.cu -o cublas_2049 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_2049 ./cublas_2049

nvcc -arch=sm_86 Cublas_gemm_4231.cu -o cublas_4231 -lcublas
sudo LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib /usr/local/cuda/bin/nsys profile --gpu-metrics-device=all --force-overwrite=true -o Cublas_A10G_4231 ./cublas_4231