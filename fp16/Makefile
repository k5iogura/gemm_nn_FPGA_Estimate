TGT := test_fp16

%.o : %.c
	gcc -o $@ -c -Ofast $^

$(TGT): fp16.o main.o
	gcc -o $@ $^ -Ofast

clean:
	rm -f *.o $(TGT)
