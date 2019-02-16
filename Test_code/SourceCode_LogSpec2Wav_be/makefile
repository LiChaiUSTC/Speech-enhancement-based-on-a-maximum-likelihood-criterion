LPS2Wav_be: LogSpec2Wav.o FEfunc.o fileio.o
	gcc -std=c99 -o LPS2Wav_be LogSpec2Wav.o FEfunc.o fileio.o -lm

LogSpec2Wav.o: LogSpec2Wav.c
	gcc -std=c99 -c -o LogSpec2Wav.o LogSpec2Wav.c -lm

FEfunc.o: FEfunc.c FEfunc.h
	gcc -std=c99 -c -o FEfunc.o FEfunc.c -lm

fileio.o: fileio.c fileio.h
	gcc -std=c99 -c -o fileio.o fileio.c -lm
     
clean:
	rm -rf LogSpec2Wav.o FEfunc.o fileio.o