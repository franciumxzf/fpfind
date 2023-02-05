.SILENT: all
.PHONY: clean
all:
	gcc -o freqcd freqcd.c getopt.c
	./generate_freqcd_testcase.py -t 0 1000000000 2000000000 |\
	   	./freqcd -f 1000000 |\
		./freqcd -f -999933 -o .output

clean:
	find . -type d -name ".pytest_cache" | xargs rm -rf
	rm -f freqcd freqcd.exe .freqcd.input
