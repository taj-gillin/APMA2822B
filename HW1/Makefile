# Compiler and Flags
CXX      = g++

# Normal
CXXFLAGS = -O2 -std=c++11

# Max Optimization
# CXXFLAGS = -O3 -std=c++11

# Fast Optimization
# CXXFLAGS = -Ofast -march=native -funroll-loops -std=c++11


# Source Files
SRCS = mv_mult_contiguous.cpp \
       mv_mult_separate_rows.cpp \
       mv_mult_loop_unrolling.cpp \
       mv_mult_padding.cpp \
       mv_mult_column_major.cpp

# Executable Names
EXES = mv_mult_contiguous \
       mv_mult_separate_rows \
       mv_mult_loop_unrolling \
       mv_mult_padding \
       mv_mult_column_major

# Default Target
all: $(EXES)

# Rule to Build Each Executable
mv_mult_contiguous: mv_mult_contiguous.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mv_mult_separate_rows: mv_mult_separate_rows.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mv_mult_loop_unrolling: mv_mult_loop_unrolling.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mv_mult_padding: mv_mult_padding.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

mv_mult_column_major: mv_mult_column_major.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean Up
clean:
	rm -f $(EXES) *.o

# Phony Targets
.PHONY: all clean
