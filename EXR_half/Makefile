TGT := test_half

LDFLAGS:= $(shell pkg-config --libs IlmBase)
CFLAGS := $(shell pkg-config --cflags IlmBase)

%.o : %.c
	g++ -o $@ -c -Ofast $^ $(CFLAGS)

$(TGT): main.o
	g++ -o $@ $^ -Ofast $(LDFLAGS)
