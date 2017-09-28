
ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g -pg
else
CXXFLAGS += -O2
endif

# Compiler
CXX ?= g++

ifeq ($(onARM),1)
CPPFLAGS+= -DonARM
endif
# Target
TARGET := gemm1.arm
TARGET_DIR := ./

EXTLIBS_DIR ?= ./extlibs

# Directories
INC_DIRS := . $(EXTLIBS_DIR)/inc
LIB_DIRS := $(EXTLIBS_DIR)/lib

# Files
INCS := $(wildcard *.h)
SRCS := $(wildcard *.c)
#LIBS := SDL2

# Make it all!
all : $(TARGET_DIR)/$(TARGET) $(OTHER_TARGETS)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : common.mk $(MK_SRCS) $(SRCS) $(INCS)
	@[ -d $(TARGET_DIR) ] || mkdir $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

aocx:gemm1.cl
	aoc --board de0_nano_sharedonly_with_spi_tft $^

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET) *.o

.PHONY : all clean
