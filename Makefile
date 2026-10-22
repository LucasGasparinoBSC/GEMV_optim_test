CXX := icpx

# Use the same core flags discussed in this workspace.
CXXFLAGS := -O3 -g -march=native -DGEMV_ENABLE_AVX512 -fp-model=precise -fno-math-errno -fno-exceptions -fno-rtti -std=c++17
# Keep vectorization diagnostics available when needed.
REPORT_FLAGS := -qopt-report=5 -qopt-report-phase=vec

TARGET := gemv
SRCS := main.cpp gemv.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean asm report

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@

%.o: %.cpp gemv.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Generate object disassembly in Intel syntax.
asm: gemv.o
	objdump -d -Mintel --no-show-raw-insn gemv.o > gemv.objdump.asm

# Rebuild with vectorization report enabled.
report:
	$(MAKE) clean
	$(CXX) $(CXXFLAGS) $(REPORT_FLAGS) -c gemv.cpp -o gemv.o
	$(CXX) $(CXXFLAGS) $(REPORT_FLAGS) -c main.cpp -o main.o
	$(CXX) $(CXXFLAGS) main.o gemv.o -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) gemv.objdump.asm gemv.optrpt
