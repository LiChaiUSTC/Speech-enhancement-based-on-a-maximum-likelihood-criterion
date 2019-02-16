#!/bin/sh
#
# dpwelibConfig.sh.in (becomes dpwelibConfig.sh through configure)
#
# This file freezes some of the configuration results of dpwelib
# so that they can be reused by subsequent dpwelib clients. 
# This is copying what the tcl install does.
# Specifically, we want to remember whether we need extra audio 
# libs on this platform.  We might as well define the path 
# to the installed header files too.  This is installed alongside 
# the libdpwe.a library file.
#
# 1997jul04 dpwe@icsi.berkeley.edu
# $Header: /u/drspeech/src/dpwelib/RCS/dpwelibConfig.sh.in,v 1.3 1997/08/13 00:30:23 dpwe Exp $

# String to pass to linker to pick up the dpwelib library from its
# installed directory, along with any other required libraries.
DPWELIB_BUILD_LIB_SPEC='-L/home/dujun/TANDEM/tools/QN/dpwelib -ldpwe '
DPWELIB_INSTALL_LIB_SPEC='-L/home/dujun/TANDEM/tools/QN/atlas1/lib -ldpwe '
DPWELIB_LIB_SPEC=${DPWELIB_INSTALL_LIB_SPEC}

# Location of the installed include headers directory dpwelib/.
DPWELIB_BUILD_INC_SPEC='-I/home/dujun/TANDEM/tools/QN/dpwelib'
DPWELIB_INSTALL_INC_SPEC='-I/home/dujun/TANDEM/tools/QN/atlas1/include'
DPWELIB_INC_SPEC=${DPWELIB_INSTALL_INC_SPEC}

# Give them the src directory so they can find Sound.[CH]
DPWELIB_SRC_DIR='/home/dujun/TANDEM/tools/QN/dpwelib'

# May as well put in this platform-specific shared library stuff too
DPWELIB_SHLIB_CFLAGS='-fPIC'
DPWELIB_SHLIB_LD=''
DPWELIB_SHLIB_SUFFIX='.so'
DPWELIB_LD_FLAGS=''
DPWELIB_LD_SEARCH_FLAGS=''

