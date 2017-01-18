GPU=0
CUDNN=0
OPENCV=0
DEBUG=0

VPATH=./
EXEC=~/locnn/liblv.so
OBJDIR=./

CC=gcc
OPTS=-O3
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors -g 

CFLAGS+=$(OPTS)

OBJ=gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o normalization_layer.o avgpool_layer.o layer.o pnet.o local_layer.o activation_layer.o batchnorm_layer.o 

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -shared -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -fPIC -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

