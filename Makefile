PWD=/home/kashim/sviluppo/
CV=opencv_3.x

EXTRA=-L/usr/local/cuda/lib64/ -L/usr/local/cuda/lib64/stubs -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_cudafeatures2d -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm -lcudart -lcublas -lcufft -lcuda -lcudart -lcudnn -lcublas -lcurand -lcusolver -I/usr/local/cuda/include/

CFLAGS=-std=c++14 `pkg-config --libs --cflags ${CV}` -I${PWD}/include/libtorch/ -I${PWD}/include/libtorch/aten/src/ -I${PWD}/include/libtorch/torch/ -I${PWD}/include/libtorch/torch/csrc/api/include/ -L${PWD}/lib/libtorch/ -ltorch_cpu -lc10 -ltorch_cuda -lc10_cuda -lpthread -lX11 -lpng

CC=g++

DCGAN=dcgan
MONO=monodepth2
MODULE=simple-module
TENSOR=simple-tensor
SSD=ssd-mobilenet
YOLO=yolov3

all:
	mkdir -p bin
	$(CC) $(DCGAN)/main.cpp $(EXTRA) $(CFLAGS) -o bin/$(DCGAN)
	$(CC) $(MONO)/main.cpp  $(EXTRA) $(CFLAGS) -o bin/$(MONO)
	$(CC) $(MODULE)/main.cpp $(EXTRA) $(CFLAGS) -o bin/$(MODULE)
	$(CC) $(TENSOR)/main.cpp $(EXTRA) $(CFLAGS) -o bin/$(TENSOR)
	$(CC) $(SSD)/main.cpp $(EXTRA) $(CFLAGS) -o bin/$(SSD)
	$(CC) $(YOLO)/main.cpp $(EXTRA) $(CFLAGS) -o bin/$(YOLO)

clean:
	rm -f *.o bin/*
