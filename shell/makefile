CANN_DIR = $(ASCEND_HOME_PATH)
OP_NAME = spmm_kernel
INCLUDE_DIR = -I../include -I../src/utils

# KERNEL_CFLAGS is still accepted for global/non-tiling experiments such as
# -DSPMM_PROFILE_AIC_ONLY=1.  Tiling defines are owned by the variant targets
# below and are filtered out to avoid accidental one-variant builds.
KERNEL_CFLAGS ?=
KERNEL_EXTRA_CFLAGS := $(filter-out -DSPMM_AIC_N_BATCH=% -DSPMM_SPARSE_K_TILE=%,$(KERNEL_CFLAGS))
HOST_CFLAGS ?= $(KERNEL_EXTRA_CFLAGS)

KERNEL_SRC = ../src/$(OP_NAME).cce
VARIANTS = \
	spmm_kernel_aic8_ktile256 \
	spmm_kernel_aic4_ktile512 \
	spmm_kernel_aic2_ktile1024 \
	spmm_kernel_aic1_ktile2048
KERNEL_OBJS = $(addprefix build/,$(addsuffix .o,$(VARIANTS)))

all: prepare $(KERNEL_OBJS) main

prepare:
	mkdir -p build

# ---- aic8 / sparse_k_tile256 ----
build/spmm_kernel_aic8_ktile256_aic.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic8_ktile256 -DSPMM_AIC_N_BATCH=8 -DSPMM_SPARSE_K_TILE=256 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic8_ktile256_aiv.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic8_ktile256 -DSPMM_AIC_N_BATCH=8 -DSPMM_SPARSE_K_TILE=256 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic8_ktile256.o: build/spmm_kernel_aic8_ktile256_aic.o build/spmm_kernel_aic8_ktile256_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	$^ \
	--allow-multiple-definition -static \
	-o $@

# ---- aic4 / sparse_k_tile512 ----
build/spmm_kernel_aic4_ktile512_aic.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic4_ktile512 -DSPMM_AIC_N_BATCH=4 -DSPMM_SPARSE_K_TILE=512 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic4_ktile512_aiv.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic4_ktile512 -DSPMM_AIC_N_BATCH=4 -DSPMM_SPARSE_K_TILE=512 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic4_ktile512.o: build/spmm_kernel_aic4_ktile512_aic.o build/spmm_kernel_aic4_ktile512_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	$^ \
	--allow-multiple-definition -static \
	-o $@

# ---- aic2 / sparse_k_tile1024 ----
build/spmm_kernel_aic2_ktile1024_aic.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic2_ktile1024 -DSPMM_AIC_N_BATCH=2 -DSPMM_SPARSE_K_TILE=1024 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic2_ktile1024_aiv.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic2_ktile1024 -DSPMM_AIC_N_BATCH=2 -DSPMM_SPARSE_K_TILE=1024 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic2_ktile1024.o: build/spmm_kernel_aic2_ktile1024_aic.o build/spmm_kernel_aic2_ktile1024_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	$^ \
	--allow-multiple-definition -static \
	-o $@

# ---- aic1 / sparse_k_tile2048 ----
build/spmm_kernel_aic1_ktile2048_aic.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic1_ktile2048 -DSPMM_AIC_N_BATCH=1 -DSPMM_SPARSE_K_TILE=2048 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic1_ktile2048_aiv.o: $(KERNEL_SRC) | prepare
	ccec -std=c++17 -c -x cce -O2 $(KERNEL_SRC) --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	$(KERNEL_EXTRA_CFLAGS) \
	-DSPMM_KERNEL_NAME=spmm_kernel_aic1_ktile2048 -DSPMM_AIC_N_BATCH=1 -DSPMM_SPARSE_K_TILE=2048 \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o $@

build/spmm_kernel_aic1_ktile2048.o: build/spmm_kernel_aic1_ktile2048_aic.o build/spmm_kernel_aic1_ktile2048_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	$^ \
	--allow-multiple-definition -static \
	-o $@

main: ../src/main.cpp $(KERNEL_OBJS)
	g++ -O2 -fpic -std=c++17 \
	$(HOST_CFLAGS) \
	../include/handle.cc \
	../src/main.cpp \
	-I$(CANN_DIR)/include \
	$(INCLUDE_DIR) \
	-L$(CANN_DIR)/lib64 \
	-o build/spmm \
	-lascendcl \
	-lruntime \
	-lstdc++ \
	-lpthread

clean:
	rm -rf build/

.PHONY: all prepare clean main
