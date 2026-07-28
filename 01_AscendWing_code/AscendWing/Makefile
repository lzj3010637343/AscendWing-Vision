# AscendWing — YOLO26 + POSIX shm IPC on Ascend 310B
# Standalone build, zero colcon/ROS/ament dependency
#
# Build:  make -j$(nproc)
# Run:    ./run.sh tracker
#         ./run.sh det_reader --benchmark

# ---- Paths ----
MX_HOME    = /root/mxVision-26.0.0
CANN_HOME  = /usr/local/Ascend/cann-9.0.0
DRV_HOME   = /usr/local/Ascend/driver

SRC_TRACKER = src/tracker
SRC_IPC     = src/ipc
BUILD_DIR   = build
INSTALL_DIR = install

# ---- Compilers & Flags ----
CXX       = g++
CC        = gcc
AR        = ar
CXXFLAGS  = -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -O2
CFLAGS    = -std=gnu11 -D_GNU_SOURCE -Wall -O2

INCLUDES  = -I$(MX_HOME)/include \
            -I$(MX_HOME)/include/MxBase \
            -I$(MX_HOME)/opensource/include/opencv4 \
            -I$(MX_HOME)/opensource/include \
            -I$(CANN_HOME)/aarch64-linux/include \
            -I$(SRC_IPC)

LIB_DIRS  = -L$(MX_HOME)/lib \
            -L$(MX_HOME)/opensource/lib \
            -L$(CANN_HOME)/aarch64-linux/lib64

RPATH     = $(MX_HOME)/lib:$(MX_HOME)/opensource/lib:$(CANN_HOME)/aarch64-linux/lib64

LDFLAGS   = $(LIB_DIRS) -Wl,-rpath,$(RPATH) -Wl,--allow-shlib-undefined \
            -lmxbase -lopencv_world -lascendcl -lglog -lpthread -lrt -ldl

# ---- Targets ----
.PHONY: all clean distclean install

all: $(INSTALL_DIR)/bin/ascendwing_tracker $(INSTALL_DIR)/bin/ascendwing_unified $(INSTALL_DIR)/bin/vision_node install/bin/detection_reader install/bin/image_reader

$(BUILD_DIR) $(INSTALL_DIR)/bin:
	@mkdir -p $@

# ---- ascend_shm static lib ----
$(BUILD_DIR)/ascend_shm.o: $(SRC_IPC)/ascend_shm.c $(SRC_IPC)/ascend_shm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libascend_shm.a: $(BUILD_DIR)/ascend_shm.o | $(BUILD_DIR)
	$(AR) rcs $@ $^

# ---- sync_frame_shm static lib (combined detection+JPEG atomic frame) ----
$(BUILD_DIR)/sync_frame_shm.o: $(SRC_IPC)/sync_frame_shm.c $(SRC_IPC)/sync_frame_shm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libsync_frame_shm.a: $(BUILD_DIR)/sync_frame_shm.o | $(BUILD_DIR)
	$(AR) rcs $@ $^

# ---- Tracker ----
$(BUILD_DIR)/mx_wrapper.o: $(SRC_TRACKER)/mx_wrapper.cpp $(SRC_TRACKER)/mx_wrapper.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_TRACKER)/main.cpp $(SRC_IPC)/ascend_shm.h $(SRC_IPC)/sync_frame_shm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(INSTALL_DIR)/bin/ascendwing_tracker: $(BUILD_DIR)/main.o $(BUILD_DIR)/mx_wrapper.o $(BUILD_DIR)/libascend_shm.a $(BUILD_DIR)/libsync_frame_shm.a | $(INSTALL_DIR)/bin
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# ---- Unified Tracker (YOLO only, SOT in Python sot_live.py) ----
$(BUILD_DIR)/main_unified.o: $(SRC_TRACKER)/main_unified.cpp $(SRC_TRACKER)/mx_wrapper.h $(SRC_IPC)/sync_frame_shm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(INSTALL_DIR)/bin/ascendwing_unified: $(BUILD_DIR)/main_unified.o $(BUILD_DIR)/mx_wrapper.o $(BUILD_DIR)/libsync_frame_shm.a | $(INSTALL_DIR)/bin
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# ---- Vision Node (yolov5cpp-style, for visual tracking) ----
$(BUILD_DIR)/vision_node.o: $(SRC_TRACKER)/vision_node.cpp $(SRC_IPC)/sync_frame_shm.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(INSTALL_DIR)/bin/vision_node: $(BUILD_DIR)/vision_node.o $(BUILD_DIR)/mx_wrapper.o $(BUILD_DIR)/libsync_frame_shm.a | $(INSTALL_DIR)/bin
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# ---- mx_model (通用 mxVision 推理库) ----
$(BUILD_DIR)/mx_model.o: $(SRC_TRACKER)/mx_model.cpp $(SRC_TRACKER)/mx_model.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libmx_model.a: $(BUILD_DIR)/mx_model.o | $(BUILD_DIR)
	$(AR) rcs $@ $^

# ---- Readers ----
$(BUILD_DIR)/detection_reader.o: $(SRC_IPC)/detection_reader.c $(SRC_IPC)/ascend_shm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/image_reader.o: $(SRC_IPC)/image_reader.c $(SRC_IPC)/ascend_shm.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(INSTALL_DIR)/bin/detection_reader: $(BUILD_DIR)/detection_reader.o $(BUILD_DIR)/libascend_shm.a | $(INSTALL_DIR)/bin
	$(CC) $(CFLAGS) $^ -Wl,-rpath,$(CANN_HOME)/aarch64-linux/lib64 $(CANN_HOME)/aarch64-linux/lib64/libascendcl.so -lpthread -lrt -ldl -o $@

$(INSTALL_DIR)/bin/image_reader: $(BUILD_DIR)/image_reader.o $(BUILD_DIR)/libascend_shm.a | $(INSTALL_DIR)/bin
	$(CC) $(CFLAGS) $^ -Wl,-rpath,$(CANN_HOME)/aarch64-linux/lib64 $(CANN_HOME)/aarch64-linux/lib64/libascendcl.so -lpthread -lrt -ldl -o $@

install: all
	@echo ""
	@echo "=== AscendWing build complete ==="
	@echo "Tracker:  $(INSTALL_DIR)/bin/ascendwing_tracker"
	@echo "Readers:  $(INSTALL_DIR)/bin/detection_reader  $(INSTALL_DIR)/bin/image_reader"
	@echo ""
	@echo "Launch:  cd /root/AscendWing && ./run.sh tracker"
	@echo ""

clean:
	rm -rf $(BUILD_DIR)/*

distclean: clean
	rm -rf $(INSTALL_DIR)/*
