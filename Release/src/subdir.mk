################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/GPUKernel.cpp \
../src/PDBReader.cpp 

OBJS += \
./src/GPUKernel.o \
./src/PDBReader.o 

CPP_DEPS += \
./src/GPUKernel.d \
./src/PDBReader.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 -gencode arch=compute_10,code=sm_10 -gencode arch=compute_13,code=sm_13 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -DRAYTRACINGENGINE_EXPORTS -DUSE_CUDA -I/usr/local/cuda/include -I/home/geek/NVIDIA_GPU_Computing_SDK/C/common/inc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


