
NVCC        = nvcc
NVCC_FLAGS  = -O3 -std=c++11
OBJ         = main.o matrix.o kernel0.o kernel1.o kernel2.o kernel3.o kernel4.o Utility.o
EXE         = spmspm


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

