#
# dictfilefns.tcl
#
# Library of fns for reading & writing Noway-style dictionaries (lexicons)
# Based on fns developed for phn2wrd, spamnotspam rewriting.
#
# 1998oct30 dpwe@icsi.berkeley.edu
# $Header: /u/drspeech/src/guitools/dpwetcl/RCS/dictfilefns.tcl,v 1.4 2000/01/22 20:08:36 dpwe Exp $

# From projects/spamnotspam/src/phn2antiphn.tcl

# Noway multiple-pronunciation dictionaries are stored in Tcl 
# arrays; basically, the element $arrayname(WORD) is 
# a list of {prior-probability {phone phone phone phone}} elements.
# However, because the WORDs can include Tcl-unfriendly characters 
# like "[" and "]", we provide GetDictDef and SetDictDef to access 
# the elements. 
#
# Note that arrays are passed by *reference* i.e. the name of the 
# array variable.

proc GetDictDef {arrayname word} {
    # Retrieve the definition of the specified word.
    # It's an error if the word is not in the dictionary
    # This cover is just to take care of the escaping of funny 
    # chrs in the word
    upvar 1 $arrayname dictArray     
    regsub -all {([]\[])} $word {\\\1} escword
    set dictent $dictArray(${escword})
    return $dictent
}

proc SetDictDef {arrayname word def} {
    # Set the definition associated with word in the dictionary
    # array.  This cover is just to handle the escaping of 
    # funny chrs in the word
    upvar 1 $arrayname dictArray     

    # Escape brackets in token (i.e. "[uh]" -> "\[uh\]")
    regsub -all {([]\[])} $word {\\\1} escword

    set dictArray($escword) $def
}

proc ReadDictNW {fname arrayname {verbose 0}} {
    # Read a Noway dictionary file into an array; 
    # the array is accessed as GetDictDef arrayname WORD which 
    # returns the list of phones read.  It can be 
    # modified with SetDictDef arrayname WORD newdef
    upvar 1 $arrayname dictArray
    # Clear it
    resetArray dictArray
    # Another array just to find existing words
    resetArray tmpWordsArray
    set f [open $fname "r"]
    set nprons 0
    set words ""
    set lasttoken ""
    while {![eof $f]} {
    	set l [string trim [gets $f]]
    	if {$l != "" && [string index $l 0] != "#"} {
	    # What word is this?
	    set token [lindex $l 0]
    	    # Remove any appended probability on token
	    set prob ""
    	    if {![regexp {^([-A-z_'*#<>.]+)(\(([-+0-9.e]+)\))?$} $token all token probclause prob]} {
    	    	puts stderr "Unable to parse line header '$token' - skipped"
    	    } else {
		if {$prob == ""} {
		    if {$verbose} {
			puts stderr "Default prob of 1.0 for token '$token'"
		    }
		    set prob 1.0
		}
		# Escape brackets in token (i.e. "[uh]" -> "\[uh\]")
		regsub -all {([]\[])} $token {\\\1} esctoken
		# Get the phone sequence
		set pron [lrange $l 1 e]
		# Save it
		#puts "lappend dictArray ($token) $pron"
		eval lappend "dictArray($esctoken)" [list [list $prob $pron]]
		incr nprons
		# Add this word to the list if it isn't already there
#    	    	if {[lsearch -exact $words $token] == -1} {
#    	    	    lappend words $token
#    	    	}
#            	if {[array names tmpWordsArray $esctoken] == ""} {
#		    lappend words $token
#		    set tmpWordsArray($esctoken) 1
#	    	}
                # ... faster by assuming prons are all contiguous
                if {$token != $lasttoken} {
		    lappend words $token
		    set lasttoken $token
		}
		# Maybe report progress
		if {$nprons > 0 && $nprons % 5000 == 0} {
		    puts stderr "(read $nprons prons from $fname)"
		}
	    }
    	}
    }
    # return number of prons read
    return [list $nprons $words]
}

proc WriteDictNW {fname arrayname {words ""}} {
    # Write the prons defined in arrayname as a Noway dictionary to fname
    # If $words is nonempty, write just those words
    upvar 1 $arrayname dictArray
    if {$words == ""} {set words [array names dictArray]}
    set f [open $fname "w"]
    set nprons 0
    foreach word $words {
    	#set dictent [lindex [array get dictArray $word] 1]
#   	regsub -all {([]\[])} $word {\\\1} escword
#   	eval set dictent \$dictArray(${escword})
	set dictent [GetDictDef dictArray $word]
    	foreach pronset $dictent {
    	    lassign $pronset prob pron
    	    puts $f "${word}(${prob}) $pron"
    	    incr nprons
	    # Maybe report progress
	    if {$nprons > 0 && $nprons % 5000 == 0} {
		puts stderr "(wrote $nprons prons to $fname)"
	    }
    	}
    }
    close $f
    return $nprons
}

