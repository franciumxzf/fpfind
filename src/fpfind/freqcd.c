/* freqcd.c : Part of the quantum key distribution software for correcting
              timestamps emitted by a timestamp unit running at a relative
              frequency offset. Proof-of-concept.

   Copyright (C) 2023 Justin Peh, Xu Zifang, Christian Kurtsiefer,
                      National University of Singapore

   This source code is free software; you can redistribute it and/or
   modify it under the terms of the GNU Public License as published
   by the Free Software Foundation; either version 2 of the License,
   or (at your option) any later version.
   This source code is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
   Please refer to the GNU Public License for more details.
   You should have received a copy of the GNU Public License along with
   this source code; if not, write to:
   Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

   --

   Program that receives a frequency offset between two parties calculated
   by fpfind, and performs a software correction of the timestamps emitted
   by readevents. Parameters are optimized such that the correction can
   be performed using purely 64-bit constructs.

   Note that there is no support for Windows - many of the functionality
   used are native to Linux.


   usage:
     freqcd [-i infilename] [-o outfilename] [-x]
            [-f freqcorr] [-F freqfilename]


   DATA STREAM OPTIONS:
     -i infilename:   Filename of source events. Can be a file or a socket
                      and has to supply binary data according to the raw data
                      spec from the timestamp unit. If unspecified, data
                      is read from stdin.
     -o outfilename:  Outfile name for timing corrected events, which can
                      either be a file or socket. Output format is the same as
                      with input. If unspecified, data is written to stdout.
     -F freqfilename: Filename of frequency correction values. Needs to be
                      a readable+writeable socket storing newline-delimited
                      frequency offset values (see '-f' option for format).
                      If unspecified, the frequency offset will be static.

   ENCODING OPTIONS:
     -x:              Specifies if both the raw input and output data streams
                      are to be read in legacy format, as specified in the
                      timestamp unit.
     -f freqcorr:     Frequency offset of the current clock relative to some
                      reference clock, in units of 2^-34 (or 0.6e-10).
                      If unspecified, offset = 0.

   Potential improvements:
     - Allow customization of frequency correction units (using 128-bit?)
     - Consider using epoll() if necessary
     - Add mechanism to handle out-of-order timestamps
     - Merge write procedure with select() call
     - Optimize buffer parameters
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>   // open, O_RDONLY, O_NONBLOCK
#include <sys/select.h> // fd_set, usually implicitly declared
#include <unistd.h>  // getopt, select
#include <errno.h>   // select errno
#include <limits.h>  // INT_MAX

/* default definitions */
#define FNAMELENGTH 200            /* length of file name buffers */
#define FNAMEFORMAT "%200s"        /* for sscanf of filenames */
#define FILE_PERMISSONS 0644       /* for all output files */
#define INBUFENTRIES 1024          /* max. elements in input buffer */
#define OUTBUFENTRIES 1024         /* max. elements in output buffer */
#define FREQBUFSIZE 100            /* max. length of frequency correction values */
#define RETRYREADWAIT 500000       /* sleep time in usec after an empty read */
#define FCORR_ARESBITS -34         /* absolute resolution of correction, in power of 2 */
#define FCORR_AMAXBITS -13         /* absolute maximum allowed correction, in power of 2 */
#define FCORR_DEFAULT 0            /* frequency correction in units of 2^FCORR_ARESBITS */
//#define __DEBUG__                /* global debug flag, uncomment to disable */

// struct defined in non-legacy format
typedef struct rawevent {
    unsigned int low;
    unsigned int high;
} rawevent;

typedef long long ll;
typedef unsigned long long ull;

/* error handling */
char *errormessage[] = {
    "No error",
    "Generic error", /* 1 */
    "Error reading in infilename",
    "Error opening input stream source",
    "Error parsing freq correction as integer",
    "Freq correction value out of range", /* 5 */
    "Unable to allocate memory to inbuffer",
    "Unable to allocate memory to outbuffer",
    "Error reading in outfilename",
    "Error opening output stream",
    "Error reading in freqfilename", /* 10 */
    "Error opening freq correction stream source",
    "Freq correction value not newline terminated",
    "Unable to allocate memory to freqbuffer",
};
int emsg(int code) {
    fprintf(stderr, "%s\n", errormessage[code]);
    return code;
};

/* for reading freqcorr values */
int readint(char *buff) {
    char *end;
    long value;
    value = strtol(buff, &end, 10);
    // string is a decimal, zero-terminated, and fits in an int
    if ((end != buff) && (*end == 0) && (abs(value) < INT_MAX))
        return (int)value;
    return INT_MAX;
}


int main(int argc, char *argv[]) {

    /* other constants */
    const int INBUFSIZE = INBUFENTRIES * sizeof(struct rawevent);
    const int OUTBUFSIZE = OUTBUFENTRIES * sizeof(struct rawevent);
    const int FCORR_MAX = 1 << (FCORR_AMAXBITS - FCORR_ARESBITS);
    const int FCORR_TBITS1 = -FCORR_AMAXBITS - 1;  // bit truncations when correcting timestamp
    const int FCORR_TBITS2 = (FCORR_AMAXBITS - FCORR_ARESBITS) + 1;

    /* parse options */
    int fcorr = FCORR_DEFAULT;  // frequency correction value
    char infilename[FNAMELENGTH] = {};  // store filename
    char outfilename[FNAMELENGTH] = {};  // store filename
    char freqfilename[FNAMELENGTH] = {};  // store filename
    int islegacy = 0;  // mark if format is legacy
    int opt;  // for getopt options
    opterr = 0;  // be quiet when no options supplied
    while ((opt = getopt(argc, argv, "i:o:F:f::x")) != EOF) {
        switch (opt) {
        case 'i':
            if (sscanf(optarg, FNAMEFORMAT, infilename) != 1) return -emsg(2);
            infilename[FNAMELENGTH-1] = 0;  // security termination
            break;
        case 'o':
            if (sscanf(optarg, FNAMEFORMAT, outfilename) != 1) return -emsg(8);
            outfilename[FNAMELENGTH-1] = 0;  // security termination
            break;
        case 'F':
            if (sscanf(optarg, FNAMEFORMAT, freqfilename) != 1) return -emsg(10);
            freqfilename[FNAMELENGTH-1] = 0;  // security termination
            break;
        case 'f':
            if (sscanf(optarg, "%d", &fcorr) != 1) return -emsg(4);
            if (abs(fcorr) >= FCORR_MAX) return -emsg(5);
            break;
        case 'x':
            islegacy = 1;
            break;
        }
    }

    /* set input and output handler */
    int inhandle = 0;  // stdin by default
    if (infilename[0]) {
        inhandle = open(infilename, O_RDONLY | O_NONBLOCK);
        if (inhandle == -1) return -emsg(3);
    }

    int outhandle = 1;  // stdout by default
    if (outfilename[0]) {
        outhandle = open(outfilename, O_WRONLY | O_CREAT | O_TRUNC, FILE_PERMISSONS);
        if (outhandle == -1) return -emsg(9);
    }

    int freqhandle = 0;  // null by default (not stdin)
    if (freqfilename[0]) {
        freqhandle = open(freqfilename, O_RDONLY | O_NONBLOCK);
        if (freqhandle == -1) return -emsg(11);
    }

    /* initialize input and output buffers */
    struct rawevent *inbuffer;
    inbuffer = (struct rawevent *)malloc(INBUFSIZE);
    if (!inbuffer) return -emsg(6);
    struct rawevent *eventptr;  // pointer to position within inbuffer
    int eventnum = 0;  // number of available rawevents for processing
    char *inbufferbytes = (char *)inbuffer;  // lower level byte buffer
    char *inbufferbytes_next;  // pointer to next byte write destination

    struct rawevent *outbuffer;
    outbuffer = (struct rawevent *)malloc(OUTBUFSIZE);
    if (!outbuffer) return -emsg(7);
    int outevents = 0;

    char *freqbuffer;
    freqbuffer = (char *)malloc(FREQBUFSIZE);
    if (!freqbuffer) return -emsg(13);
    int freqbytesread = 0;
    int freqbytesread_next = 0;
    int freqbytespartial = 0;  // size of partial freqcorr value remaining
    char *freqbuffer_next = freqbuffer;  // pointer to next char write destination

    /* parameters for select call */
    fd_set rfds;
    struct timeval tv;
    int retval;

    /* inbuffer reading variables */
    int i, j;
    int inbytesread = 0;
    int inbytesread_next = 0;
    int inbytespartial;  // size of partial rawevent remaining in inbufferbyte

    /* timestamp variables */
    ull tsref = 0;  // reference timestamp to scale by frequency correction,
                    // noting subsequent initializations should zero 'tsref'
    int isset_tsref = 0;    // initialization marker for tsref
    ull ts;                 // timestamp
    ll tscorr;              // timestamp correction
    ll tsoverflowcorr = 0;  // timestamp overflow corrections
    unsigned int high;      // high word in timestamp
    unsigned int low;       // low word in timestamp
    unsigned int _swp;      // temporary swap variable, support 'legacy' option

    /* main loop */
    while (1) {

        /* discard previously processed rawevents and
           retain partial rawevent left in buffer */
        inbytespartial = inbytesread % sizeof(struct rawevent);
        for (i = inbytesread - inbytespartial, j = 0; j < inbytespartial; i++, j++) {
            inbufferbytes[j] = inbufferbytes[i];
        }
        inbufferbytes_next = &inbufferbytes[inbytespartial];

        /* wait for data on inhandle and freqhandle */
        // TODO: Consider whether to use poll/epoll mechanisms, if frequent
        //       pipe recreation is a concern (high fd).
        FD_ZERO(&rfds);
        FD_SET(inhandle, &rfds);
        if (freqhandle) FD_SET(freqhandle, &rfds);
        tv.tv_sec = 0;
        tv.tv_usec = RETRYREADWAIT;
        retval = select(FD_SETSIZE, &rfds, NULL, NULL, &tv);
        if (retval == -1) {
            fprintf(stderr, "Error %d on select", errno);
            break;  // graceful close
        }

        if (FD_ISSET(inhandle, &rfds)) {

            /* read data from inhandle */
            // TODO: Highlight corresponding bug in chopper.c. Note that assigning
            //       to inbytesread directly can potentially corrupt events.
            inbytesread_next = read(inhandle, inbufferbytes_next, INBUFSIZE - inbytespartial);
            if (inbytesread_next == 0) {
                break;  // no bytes read (i.e. EOF)
                        // TODO: Check if this should be continue instead,
                        //       when running ad-infinitum
            }
            if (inbytesread_next == -1) {
                fprintf(stderr, "Error %d on read", errno);
                break;  // graceful close
            }

            /* concatenate new data */
            inbytesread = inbytespartial + inbytesread_next;
            eventnum = inbytesread / sizeof(struct rawevent);
            eventptr = inbuffer;

            /* micro-optimization to initialize reference timestamp */
            if ((!isset_tsref) && (eventnum > 0)) {
                low = eventptr->low;
                high = eventptr->high;

                // Shift burden of swapping if 'legacy' format is used
                // TODO: Consider a more efficient implementation.
                if (islegacy) {
                    _swp = low;
                    low = high;
                    high = _swp;
                }
                tsref = ((ull)high << 22) | (low >> 10);
                isset_tsref = 1;  // we are done initializing
            }

            /* digest events */
            for (i = 0; i < eventnum; i++) {

                /* extract timestamp value */
                // Assumed 4ps timestamps used
                low = eventptr->low;
                high = eventptr->high;
                if (islegacy) {
                    _swp = low;
                    low = high;
                    high = _swp;
                }
                ts = ((ull)high << 22) | (low >> 10);
#ifdef __DEBUG__
                fprintf(stderr, "[debug] Raw event - %08x %08x\n", high, low);
                fprintf(stderr, "[debug] |   t_i: %014llx (%020llu)\n", ts, ts);
#endif

                /* calculate timestamp correction */
                tscorr = ((ll)((ts - tsref) >> FCORR_TBITS1) * fcorr) >> FCORR_TBITS2;
                ts += tscorr;

                /* account for 20 hour timestamp overflow condition */
                // TODO: Need to account for the 20 hour timestamp overflow condition
                //       which might be a bit messy due to non-monotonically increasing
                //       timestamps as a result of say, detector timing corrections.
                //       Consider the following corner cases:
                //         1. If curr < prev, then prev-curr > 10 hours => overflow.
                //            Save tsref and set to zero, overflow++.
                //         2. If curr > prev, then curr-prev > 10 hours => underflow.
                //            Restore previous tsref, overflow--;

                /* write corrected timestamp to output buffer */
                eventptr->high = ts >> 22;
                eventptr->low = (ts << 10) | (low & 0x3ff);
#ifdef __DEBUG__
                fprintf(stderr, "[debug] |  t'_i: %014llx (%020llu)\n", ts, ts);
                fprintf(stderr, "[debug] +---------- %08x %08x\n", eventptr->high, eventptr->low);
#endif
                if (islegacy) {
                    _swp = eventptr->low;
                    eventptr->low = eventptr->high;
                    eventptr->high = _swp;
                }
                outbuffer[outevents++] = *eventptr;
                eventptr++;
            }
        }

        // Read frequency correction values
        // - Note partial buffer must be maintained, since there is no other
        //   check to verify integrity of values broken up between separate reads.
        // - Note also this falls after reading the input buffer, but there is no
        //   particular reason why this order is chosen.
        if (freqhandle && FD_ISSET(freqhandle, &rfds)) {
            freqbytesread_next = read(freqhandle, freqbuffer_next, FREQBUFSIZE - freqbytespartial - 1);

            // File/pipe closed -> proceed without any further fcorr updates
            if (freqbytesread_next == 0) {
                freqhandle = 0;
#ifdef __DEBUG__
                fprintf(stderr, "[debug] File/pipe '%s' closed.\n", freqfilename);
                // hidden in buffer to avoid debug leak
                // no break here
#endif
            }
            if (freqbytesread_next == -1) {
                fprintf(stderr, "Error %d on freqhandle read.\n", errno);
                break;
            }

            /* concatenate new data */
            freqbytesread = freqbytespartial + freqbytesread_next;

            /* search for valid fcorr values */
            int next_num_idx = 0, fcorr_tmp;
            for (i = 0; i < freqbytesread; i++) {
                if (freqbuffer[i] == '\n') {
                    freqbuffer[i] = 0;  // zero-terminate for readint
                    fcorr_tmp = readint(&freqbuffer[next_num_idx]);
                    if (abs(fcorr_tmp) < FCORR_MAX) fcorr = fcorr_tmp;
#ifdef __DEBUG__
                    fprintf(stderr, "[debug] 'fcorr' updated to '%d'.\n", fcorr);
#endif
                    next_num_idx = i+1;  // continue reading
                }
            }

            /* clear parsed numbers from freqbuffer */
            if (next_num_idx > 0) {
                freqbytespartial = freqbytesread - next_num_idx;
                for (i = freqbytesread - freqbytespartial, j = 0; j < freqbytespartial; i++, j++) {
                    freqbuffer[j] = freqbuffer[i];
                }
                freqbuffer_next = &freqbuffer[freqbytespartial];
            }
        }

        // TODO: Shift this back to select call to write only when pipe available,
        //   and increase buffer size of output pipe in case write unavailable.
        //   By same measure, do not flush only when output buffer is full.
        /* write out events */
        retval = write(outhandle, outbuffer, eventnum * sizeof(struct rawevent));
#ifdef __DEBUG__
        for (i = 0; i < eventnum; i++) {
            fprintf(stderr, "[debug] Verify: %08x %08x\n", outbuffer[i].high, outbuffer[i].low);
        }
#endif
        if (retval != eventnum * sizeof(struct rawevent)) {
            fprintf(stderr, "Error %d on write", errno);
            break;  // graceful close
        }
        outevents = 0;  // clear outbuffer only after successful write
        eventnum = 0;  // clear events to avoid rewriting:
                       // occurs when 'freqhandle' available for reading, but
                       // 'inhandle' has no more events
    }

    /* free buffers */
    free(inbuffer);
    free(outbuffer);
    free(freqbuffer);
    return 0;
}
