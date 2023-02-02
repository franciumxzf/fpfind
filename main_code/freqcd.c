/*
    usage: freqcd -i infilename
                  [-f freqcorrection]
*/

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#ifdef _WIN32
    #include "getopt.h"
    // Consider the full getopt port:
    // https://www.codeproject.com/Articles/157001/Full-getopt-Port-for-Unicode-and-Multibyte-Microso
#else
    #include <unistd.h>  // getopt
#endif

/* default definitions */
#define FNAMELENGTH 200            /* length of file name buffers */
#define FNAMEFORMAT "%200s"        /* for sscanf of filenames */
#define INBUFENTRIES 1024          /* max. elements in input buffer */
#define RETRYREADWAIT 500000       /* sleep time in usec after an empty read */
#define DEFAULTFCORR 0             /* default frequency correction in 2^-34 units */
#define RESFCORR -34               /* resolution of correction, in power of 2 */
#define MAXFCORR -13               /* maximum allowed correction, in power of 2*/

typedef struct rawevent {
    unsigned int cv, dv;
} re;

typedef long long ll;
typedef unsigned long long ull;

/* error handling */
char *errormessage[] = {
    "No error",
    "Generic error", /* 1 */
    "Error reading in infilename",
    "Error opening input stream source",
    "Error parsing freq correction as integer",
    "Freq correction value out of range",
};
int emsg(int code) {
    fprintf(stderr, "%s\n", errormessage[code]);
    return code;
};

int main(int argc, char *argv[]) {
    int opt;
    char *pipe;
    int fcorr = DEFAULTFCORR;
    int remfcorr = MAXFCORR-RESFCORR+1;
    
    char infilename[FNAMELENGTH] = {};
    int inhandle = 0;  /* stdin by default */
    int inbytesread;
    char *inbuf;

    /* parse options */
    opterr = 0;  // be quiet when no options supplied
    while ((opt = getopt(argc, argv, "i:f::")) != EOF) {
        switch (opt) {
        case 'i':
            if (sscanf(optarg, FNAMEFORMAT, infilename) != 1) return -emsg(2);
            infilename[FNAMELENGTH-1] = 0;
            break;
        case 'f':
            if (sscanf(optarg, "%d", &fcorr) != 1) return -emsg(4);
            if (abs(fcorr) >= (1 << (MAXFCORR-RESFCORR))) return -emsg(5);
            break;
        }
    }

    /* set input handler */
    // TODO(Justin, 2023-02-02):
    //   Windows port is messy - O_NONBLOCK is not a defined macro, since
    //   the way streams work is fundamentally different, see overlapped I/O:
    //   https://learn.microsoft.com/en-us/windows/win32/fileio/synchronous-and-asynchronous-i-o
    if (infilename[0]) {
        inhandle = open(infilename, O_RDONLY);  // | O_NONBLOCK);
        if (inhandle == -1) return -emsg(3);
        // inbytesread = read(inhandle, inbuf, 1);
    }
    
    /* run parser */
    while (1) {
        // TODO
        /* Example below where ts = 1,000,000,000:
         * ./freqcd -f 1000000
         *   => actual df = 1_000_000 * 2^-34 = 0.00005820766
         *   => tscorr = 58,207
         *   => ts' = 1,000,058,207
         */

        /* read timestamp */
        ull ts = 1e9;

        /* mark first timestamp as start */
        ull startts = 0;

        /* correct timestamp */
        ll tscorr = (((ts - startts) >> (-MAXFCORR-1)) * fcorr) >> remfcorr;
        
        /* add timestamp */
        ts += tscorr;
        printf("%ld\n", ts);
        break;
    }
    return 0;
}