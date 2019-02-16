Wav2LPS_be: Wav2LPS_be.o FEfunc.o fileio.o
	gcc -std=c99 -o Wav2LPS_be Wav2LPS_be.o FEfunc.o fileio.o -lm

Wav2LPS_be.o: Wav2LogSpec_be.c
	gcc -std=c99 -c -o Wav2LPS_be.o Wav2LogSpec_be.c -lm

FEfunc.o: FEfunc.c FEfunc.h
	gcc -std=c99 -c -o FEfunc.o FEfunc.c -lm

fileio.o: fileio.c fileio.h
	gcc -std=c99 -c -o fileio.o fileio.c -lm
     
clean:
	rm -rf Wav2LPS_be.o FEfunc.o fileio.o