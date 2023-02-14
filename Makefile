# Define the target name
TARGET = run

# Define the CUDA compiler
CUDA = nvcc

# Define the CUDA flags
CUDA_FLAGS = -std=c++11

# Define the object files
OBJS =	matoper.o	#tensor.o	

# Define the target
$(TARGET): $(OBJS)
	$(CUDA) -o $(TARGET) $(OBJS)

# Define the matoper.o object file rule
matoper.o: matoper.cu tensor.h
	$(CUDA) $(CUDA_FLAGS) -c matoper.cu

# Define the tensor.o object file rule
# tensor.o: tensor.cu tensor.h
# 	$(CUDA) $(CUDA_FLAGS) -c tensor.cu

# Define the clean rule
clean:
	rm -f $(TARGET) $(OBJS)

##header 파일 에러 수정하기