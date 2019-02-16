/* config.h.  Generated automatically by configure.  */
/* config.h.in.  Generated automatically from configure.in by autoheader.  */

/* Define if type char is unsigned and you are not using gcc.  */
#ifndef __CHAR_UNSIGNED__
/* #undef __CHAR_UNSIGNED__ */
#endif

/* Define if you have the ANSI C header files.  */
#define STDC_HEADERS 1

/* Define if your processor stores words with the most significant
   byte first (like Motorola and SPARC, unlike Intel and VAX).  */
/* #undef WORDS_BIGENDIAN */

/* Define if you have the hypot function.  */
#define HAVE_HYPOT 1

/* Define if you have the isatty function.  */
#define HAVE_ISATTY 1

/* Define if you have the popen function.  */
#define HAVE_POPEN 1

/* Define if you have the strdup function.  */
#define HAVE_STRDUP 1

/* Define if you have the <sys/audioio.h> header file.  */
/* #undef HAVE_SYS_AUDIOIO_H */

/* Define if you have the <unistd.h> header file.  */
#define HAVE_UNISTD_H 1
/*************** -*- C -*- ******************************\
*  dpwelib.h.in						*
*  little header to try and setup STDC for my lib	*
*  dpwe 28may91						*
\*******************************************************/

/* WHAT'S HERE:  a bunch of stuff typically needed for my code
                 - but you might not need it all
 Includes (& what they are needed for)
   * <stdio.h>	(printf, FILE, stderr, fopen, (size_t))
   * <stdlib.h> or equivalent (malloc, atof, labs, NULL, size_t)
   * <fcntl.h>	(O_RDONLY etc)   		 
   * <string.h>	(strcmp, rindex, strrchr)
 Defines
   * READMODE, WRITEMODE for fopen of binary files
   * FLOATARG for type of floats in prototypes
 Macros
   * TMMALLOC(type, count, errmsg) allocates a pointer to <count> of <type>
   * MIN, MAX (arguments evaluated twice)
   * PARG -- for optional argument prototypes : void fn PARG((type arg));
 PARG is pretty important, but you could include it explicitly 
 in any stand-alone header you wanted to build.
   * DBGFPRINTF(()) - like fprintf *except* must use double brackets, 
     only does anything if -DDEBUG
 */

#ifndef _DPWELIB_H_
#define _DPWELIB_H_

/* autoconf stuff */
/* #include <config.h> */
/* dpwelib.h is now built from this file AND config.h */

/* Standard includes */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
/* sun4 needs <unistd.h> for fseek codes (SEEK_SET etc) */
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

/* Prototype argument wrapper */
/* make fn protos like   void fn PARG((int arg1, char arg2));  */
#ifdef __STDC__
#define PARG(a)		a
#else /* !__STDC__ */
#define PARG(a)		()
#endif /* __STDC__ */

/* 2nd try - setup PARG for C++ too */
#ifdef __cplusplus
#undef PARG
#define PARG(a)		a
#endif /* __STDC__ */

/* some general utilities to put in a .h file */
#ifndef MIN
#define MIN(a,b)	(((a)<(b))?(a):(b))
#define MAX(a,b)	(((a)>(b))?(a):(b))
#endif /* !MIN */

/* from genutils.h .. need them earlier! */
char *Mymalloc PARG((long num, long siz, char *title));
void Myfree PARG((char *ptr, char *title));

#define TMMALLOC(type,num,msg)	(type *)Mymalloc(num, sizeof(type), msg)
#define TMFREE(ptr,msg)		Myfree((char *)ptr,msg)

#ifdef DEBUG
#define DBGFPRINTF(a)	fprintf a
#define FPRINTF(a)	fprintf a
#ifdef THINK_C
#define ERRTOFILE
#define ABORT		Debugger()
#else  /* DEBUG, !THINK_C */
#define ABORT		abort()
#endif /* THINK_C */
#ifdef ERRTOFILE
#undef stderr
#define stderr	((errfile==NULL)?(errfile=fopen("stderr.log","w")):errfile)
extern FILE *errfile;
#endif /* ERRTOFILE */
#else /* !DEBUG */
#define DBGFPRINTF(a)	/* no DBGFPRINTFs if not debug */
#ifdef THINK_C
#define FPRINTF(a)	/* also, no fprintfs at all on mac */
#define ABORT		return 0	/* .. lame excuse for abort */
#else /* !THINK_C */
#define FPRINTF(a)	fprintf a
#define ABORT		abort()
#endif /* THINK_C */
#endif /* DEBUG */

#ifndef HAVE_HYPOT
#define	hypot(a,b)	sqrt(pow(a,2.0)+pow(b,2.0))
#endif /* !HAVE_HYPOT */

/* hmm.. generally useful types? */
#define FLOATARG	double	/* for arguments - promotes anyway */

#ifndef INT32
#ifndef INT32_DEFINED
#define INT32	int
#define INT32_DEFINED
#endif /* INT32_DEFINED */
#endif /* INT32 */

#ifndef PI
#ifdef M_PI
#define PI M_PI
#else  /* M_PI */
#define PI	3.14159265358979323846
#endif /* M_PI */
#endif /* PI */

#ifndef TWOPI
#define TWOPI 	(2.0*PI)
#endif /* TWOPI */

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#endif /* _DPWELIB_H_ */
