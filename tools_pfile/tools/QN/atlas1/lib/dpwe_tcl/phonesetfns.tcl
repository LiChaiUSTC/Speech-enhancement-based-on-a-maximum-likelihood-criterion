#
# phonesetfns.tcl
#
# Tcl scripts for manipulating phoneset files
#
# 1997mar25 dpwe@icsi.berkeley.edu
# $Header: /n/abbott/da/drspeech/src/guitools/dpwetcl/RCS/phonesetfns.tcl,v 1.8 2001/08/19 22:11:54 dpwe Exp $

set _phone_beglabel "*B"
set _phone_endlabel "*E"

proc ReadPhsetFile {filename} {
    # Read in a definition of a phoneset; return a list of the symbols
    set psf [open $filename "r"]
    # First line is number of phonemes
    set nn [gets $psf]
    # Remaining lines are phoneme definitions
    set labels {}
    for {set i 0} {$i < $nn} {incr i} {
	lappend labels [lindex [gets $psf] 0]
    }
    close $psf
    return $labels
}

proc ReadPhoneset {filename} {
    # Read a phoneset file & set up the phonesetLabels global
    # Return number of labels read
    global phonesetLabels phonesetArray phonesetFile 
    global _phone_beglabel _phone_endlabel
    catch {unset phonesetArray}
    set phonesetArray($_phone_beglabel) -1
    set phonesetArray($_phone_endlabel) -2
    set phonesetLabels {}
    set file [open $filename "r"]
    set nphones [gets $file]
    for {set i 0} {$i < $nphones} {incr i} {
	#lassign [gets $file] label index
	set l [gets $file] 
	set label [lindex $l 0]
	set index [lindex $l 1]
	# Check for syntax
	if {$index != $i} {
	    Warn "Item $i in phonesetfile $filename is out of sequence: '$l'"
	}
	lappend phonesetLabels $label
	set phonesetArray($label) $index
    }
    close $file
    set phonesetFile $filename
    return $nphones
}

proc PhoneName {index} {
    # Return a string with which to label a state number
    global phonesetLabels _phone_beglabel _phone_endlabel
    if {$index == -1} {return $_phone_beglabel}
    if {$index == -2} {return $_phone_endlabel}
    if {[llength $phonesetLabels] > $index} {
	return [lindex $phonesetLabels $index]
    } else {
	return $index
    }
}

proc PhoneIndexOld {name} {
    # Return a numerical index corresponding to a state number
    global phonesetLabels _phone_beglabel _phone_endlabel
    if {$name == $_phone_beglabel} {return -1}
    if {$name == $_phone_endlabel} {return -2}
    set index [lsearch -exact $phonesetLabels $name]
    if {$index > -1} {
	return $index
    }
    # Looking bad.  If it's already a number, return it as is
    if {[regexp {^[0-9]*$} $name]} {
	return $name
    }
    # Couldn't figure it at all
    puts stderr "PhoneIndex: cannot find \"$name\""
    return -99
}

proc PhoneIndex {name} {
    # Return a numerical index corresponding to a state number
    global phonesetArray
    if {[catch {set ix $phonesetArray($name)} rslt]} {
	puts stderr "PhoneIndex: cannot find \"$name\" ($rslt)"
	return -99
    }
    return $ix
}

proc Setup61to56 {} {
    # Configure the array to perform the timit->icsi mapping
    # In the icsi-56 context, running this then using 
    # PhoneIndex will give the right output numbers even for
    # the timit61 phoneset
    global phonesetArray
    set mappings {{eng ng} {ux uw} {epi h#} {pau h#} {ax-h ax}}
    foreach pair $mappings {
	lassign $pair from to
	set phonesetArray($from) $phonesetArray($to)
    }
}

proc Setup61to54 {} {
    # Configure the array to perform the timit->cam mapping
    # In the cam54 context, running this then using 
    # PhoneIndex will give the right output numbers even for
    # the timit61 phoneset (or ICSI56!)
    global phonesetArray
    set mappings {{eng ng} {ux uw} {epi h#} {pau h#} {ax-h ax} {q h#} {nx n}}
    foreach pair $mappings {
	lassign $pair from to
	set phonesetArray($from) $phonesetArray($to)
    }
}

# Default phoneset
set phonesetFile "/u/drspeech/data/phonesets/icsi56.phset"
