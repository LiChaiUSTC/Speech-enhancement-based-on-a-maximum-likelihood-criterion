#
# syllifyfns.tcl
#
# Tcl scripts for talking to tsylb for syllabification and manipulating data
# Based on various STP and other procedure sets.
#
# Written by dpwe@icsi.berkeley.edu
# 1998oct30 extracted from syllify by fosler@icsi.berkeley.edu
#
# $Header: /u/drspeech/src/guitools/dpwetcl/RCS/syllifyfns.tcl,v 1.2 1998/11/02 05:05:11 dpwe Exp $
 
# Package header
set syllifyfns_vsn 0.1
package provide "Dpwe_Syllifns" $syllifyfns_vsn

proc Syllify_Init {} {
    # Dummy fn to force loading of pkg
}

proc OpenTsylb {{my_tsylb_bin ""} {my_tsylb_args ""}} {
    # Instantiate the pipe to the child tsylb program
    # Store the handle in global sylpipe
    # $sylpipe is initialized in initialization block at bottom of file
    global sylpipe tsylb_bin tsylb_args

    if {$my_tsylb_bin == ""} {set my_tsylb_bin $tsylb_bin}
    if {$my_tsylb_args == ""} {set my_tsylb_args $tsylb_args}

    if {$sylpipe == ""} {
	if {![file executable $my_tsylb_bin]} {
	    puts stderr "Couldn't find tsylb binary at $my_tsylb_bin - please set TSYLBBIN & retry"
	    return -1
	} else {
	    set odir [pwd]
	    set tdir [file dirname $my_tsylb_bin]
	    set tstem [file tail $my_tsylb_bin]
	    cd $tdir
	    set sylpipe [open "|./$tstem $my_tsylb_args" "r+"]
	    cd $odir
	    if {$sylpipe == ""} {
		puts stderr "Error opening $my_tsylb_bin"
		return -1
	    }
	    # Otherwise, just check we're getting the TSYLB banner
	    set done 0
	    while {!$done} {
		set line [gets $sylpipe]
		if {[string match "*TSYLB*" $line] || [eof $sylpipe]} {
		    set done 1
		}
	    }
	    if {[eof $sylpipe]} {
		close $sylpipe
		set sylpipe ""
		puts stderr "Got EOF from $my_tsylb_bin before startup banner"
		return -1
	    }
	}
    }
    return 0
}

proc syllify {phons {id "????-?-????"}} {
    # Takes a string of phons and returns the same phons but 
    # arranged into lists for each phoneme i.e. 
    # input = "d ih m axr"
    # output = "{d ih} {m axr}"
    # $id is just used for error messages.

    # sylpipe is the fileid under which tsylb is running
    global sylpipe xs2next
    puts $sylpipe $phons
    flush $sylpipe
    # Read back the stdout of the process until we get the "Basic pron" line
    while {[string match "*Basic pron*" [set line [gets $sylpipe]]]==0} {
	# Report error lines
	if {[string match "*\*ERR*" $line]} {
	    puts stderr "Error: $line"
	    puts stderr "   in: $id $phons"
	    return ""
	}
	# keep looping
    }
#    set line { Basic pron is /# [ b '1 iy [ t ] '0 er ] #/}
    # Ok, now we have the line.  Strip out the phon sequences
    regexp {/#(.*)#/} $line all sqce
    # Strip out the asterisks
    regsub -all {\* } $sqce "" sqce
    # Strip out other stress markings
    regsub -all {'[0-9] } $sqce "" sqce
    # trim external whitespace
    set sqce [string trim $sqce]
    # Further wrinkles: tsylb may leave certain phonemes out of any syllables -
    # (i.e. [ ay t ] k [ t ih n ]).  Add to preceding or following syllable
    # based on $xs2next (find them first for verification)
    if {[string trim [regmod -all {\[[^]]*\]} $sqce ""]] != ""} {
	set s2 $sqce
	regsub -all {\[ } $s2 \{\{ s2
	regsub -all { \]} $s2 \}\} s2
#	puts stderr "** Will fix \"$s2\""
	set six 0
	foreach s $s2 {
	    if {[string range $s 0 0] != "\{"} {
		set substr [lrange $s2 [expr $six-2] [expr $six+2]]
		regsub -all \{\{ $substr \{ substr
		regsub -all \}\} $substr \} substr
		puts "$id ** extrasyllabic: $substr"
	    }
	    incr six
	}
    }
    # Now actually reintegrate them...
    if {$xs2next} {
	# Add inter-syllable extrasyllabics to following syllable
	regsub -all { ([^]]+) \[} $sqce " \[ \\1" sqce
    } else {
	# Add inter-syllable extrasyllabics to end of preceding syllable
	regsub -all {\] ([^[]+) } $sqce "\\1 \] " sqce
    }
    # Also handle "k [t ih n]" (stray consonant at start)
    regsub -all {^([^[]+) \[} $sqce "\[ \\1" sqce
    # Also handle "[ay t] k" (stray consonant at end)
    regsub -all {\] ([^[]+)$} $sqce "\\1 \]" sqce

    # convert " ]" to close-brace and "[ " to open-brace
    if {[set nop [regsub -all { \]} $sqce \} sqce]]==0} {
	# no braces found in the data - must be a single consonant
	return "\{[string trim $sqce]\}"
    }
    set ncl [regsub -all {\[ } $sqce \{ sqce]
    if {$nop != $ncl} {
	puts stderr "$id ## OPEN/CLOSE mismatch in \"$sqce\""
	# A hack - clear the first brace we find; for singletons, will fix
	regsub {[\{\}]} $sqce "" sqce
    }

    # Ambisyllabic phonemes are marked by [d ih [m] ix r] - 
    # assign to 2nd syll
    foreach syl $sqce {
	if {[set bpos [string first \{ $syl]] >= 0} {
	    # If this syllable contains a \{ and \} (i.e. nested list)
	    # break it there to make two pieces;  one everything up 
	    # to the \{, and the other the remainder with \}'s removed
	    set ix [lsearch $sqce $syl]
	    set pt1 [string range $syl 0 [expr $bpos-1]]
	    set pt2 [string range $syl [expr $bpos+1] e]
	    regsub \} $pt2 "" pt2
	    set sqce [lreplace $sqce $ix $ix $pt1 $pt2]
	}
    }
    # OK
    return $sqce
}



# ------- rewrite 1997may29 to carry across diacritics ------

proc PreprocessPhns {labs {id "????-?-????"} {keepdias 0} {mergecls 0}} {
    # Rewrite of the pre-tsylb filtering step.  This 
    # time, work on the durlab format {btime dur tok} {btime dur tok}
    # and optionally re-attach diacritics after merging (if $keepdias)
    set oplabs ""
    # Add special end symbol
    set elabs [concat $labs "{0 0 *END*}"]
    # Tolerate inaccuracies this big in start/end alignments
    set sloptime 0.001
    # Figure the stripped form of this sequence for reporting errors
    set errname "$id ([lmap $labs {nth 2}])"
    # Initialize loop variables
    set lasttok ""
    set lastbeg 0.0
    set lastdur 0.0
    set lastdia ""
    set lastdias ""
    foreach lab $elabs {
	lassign $lab beg dur tok
	# Split off any dias
	set dia ""
	regexp {([^_]*)_(.*)} $tok all tok dia
	# Convert diacritics into a list
	set dias [split $dia "_"]
	# ** Special case modification:
	# Collapse xcl .. x to one x by mapping xcl -> x
	if {[regsub {cl$} $tok "" tok]} {
	    # Note it in the dia for match-warning purposes
	    set dia [join [concat "cl" $dia] "_"]
	}
	# Before we merge two identical labels, report the error
	if {$lasttok == $tok && $lastdia == $dia} {
	    Warn "**prePhns: not merging repeated [join [concat $tok $dia] _] in $errname"
	}
#puts "$lasttok $tok [expr $lastbeg+$lastdur+$sloptime] $beg"
	# If this token was deleted, treat it as continuation
	if {$tok == ""} {
	    Warn "**prePhns: null tok in $lab, $errname"
	}
	# Check $dia != $lastdia so identical repeats don't merge
	if {($tok == "" \
		|| ($mergecls && [string match "cl*" $lastdia])
		|| ($lasttok == $tok && $lastdia != $dia)) \
		&& $beg <= ($lastbeg+$lastdur+$sloptime)} {
	    # "Xcl" merges with any following symbol, but warn if mismatch
	    # This is kind of a problem for burst-less final closures, 
	    # but we have to live with it, I think.
	    if {[string match "cl*" $lastdia]} {
		if {$tok != $lasttok} {
		    Warn "$id: ${lasttok}cl merged to $tok"
		    set lasttok $tok
		}
		# in any case, the cl just merges to the next, then
		# gets eaten up
		regsub "^cl(_?)" $lastdia "" lastdia
	    }
	    # Continuation of previous label with altered diacritic 
	    # and continuity of time.
	    # Keep the orignal beginning, but extend duration and dias
	    set lastdur [expr $lastdur + $dur]
	    # Only take new dias
	    set lastdias [lunion $lastdias $dias]
	} else {
	    if {$lasttok != ""} {
		# Token change - emit the one we're waiting on
		if {$keepdias} {
		    set optok [join [concat $lasttok $lastdias] "_"]
		} else {
		    set optok $lasttok
		}
		lappend oplabs [list $lastbeg $lastdur $optok]
	    }
	    # Record the new token as the one to extend
	    set lasttok $tok
	    set lastdur $dur
	    set lastbeg $beg
	    # $lastdia is the actual diacritic string on the last token
	    #  (it also includes any "cl" part split).  It is 
	    #  used to avoid merging exact replica phones
	    # $lastdias is the list of split-apart diacritics 
	    #  accumulated throughout the current token.
	    set lastdia $dia
	    set lastdias $dias
	}
    }
    return $oplabs
}

proc filterPhns {durlabs {maplist ""} {warns ""} {id "????-?-????"}} {
    # Rebuilt a durlab kind of phnlist, applying a set of regsub
    # mappings from $maplist, and reporting warnings if glob patterns
    # in $warns match - applied to the tok field
    # $maplist is a list of {from to} pairs for regsubbing 
    # the tokens (to do mappings) (e.g. {{"^axr" "er"} {"_gl" "_lg"}})
    # $warns is a list of glob patterns to pick out tokens that 
    # should generate warnings (e.g. {"*_tr*"})
    set olabs ""
    foreach lab $durlabs {
	lassign $lab beg dur tok
	# Check for tokens matching warn patterns
	foreach warn $warns {
	    if {[string match $warn $tok]} {
		Warn "**filterPhns: token $tok matches warn $warn in $id"
	    }
	}
	# Perform mappings
	foreach map $maplist {
	    lassign $map from to
	    regsub $from $tok $to tok
	}
	lappend olabs [list $beg $dur $tok]
    }
    return $olabs
}

proc SyllabifyLabels {durlabs {id "????-?-????"} {keepdias 0} {mergecls 0}} {
    # Run the syllabifier on durlab-format labels using the 
    # new preprocessor
    global map61to56
    # Run the token mappings & check for warnings
    # Token map is a series of regsub {from to} pairs
    set tokenmap {{UNL ""} {unl ""} {^\\? ""} {^! ""} {SIL "h#"} {sil "h#"} \
	    {SP "h#"} {sp "h#"}}
    if {$map61to56} {
	set tokenmap [concat $tokenmap {{eng ng} {ux uw} {epi h#} {pau h#} {ax-h ax}}]
    }

    # warns is a set of "glob" patterns which generate a warning msg to 
    # stderr if seen
    set warns {}
    set flabs [filterPhns $durlabs $tokenmap $warns $id]
    # Preprocess the phonemes (merge repeats)
    set mlabs [PreprocessPhns $flabs $id $keepdias $mergecls]
    # Break up into parallel lists - begs, durs, phns, dias
    set begs ""
    set durs ""
    set phns ""
    set dias ""
    foreach mlab $mlabs {
	lassign $mlab beg dur tok
	regexp {([^_]*)(.*)} $tok all phn dia
	lappend begs $beg
	lappend durs $dur
	lappend phns $phn
	lappend dias $dia
    }
    # Initialize output
    set olabs ""
    set pos 0
    set plen [llength $phns]
    # Loops on the h# separator
    while {$pos < $plen} {
	# Pass everything up to the next h#
	set sqlen [lsearch "[lrange $phns $pos e] h#" "h#"]
	set sqphns [lrange $phns $pos [expr $pos+$sqlen-1]]
	# Syllabify the phn list (base phns)
	if {$sqphns != ""} {
	    set syls [syllify $sqphns $id]
	    # Returns "" on error
	    if {$syls == ""} {return ""}
	} else {
	    set syls ""
	}
	# Go through creating appropriate labels for each phn	    
	foreach syl [concat $syls "h#"] {
	    set sylbeg [lindex $begs $pos]
	    set syldur 0.0
	    set syltok ""
	    foreach phn $syl {
		if {$pos < $plen} {
		    Assert "\\\"$phn\\\" == \\\"[lindex $phns $pos]\\\""
		    set syldur [expr $syldur + [lindex $durs $pos]]
		    lappend syltok [lindex $phns $pos][lindex $dias $pos]
		    incr pos
		}
	    }
	    if {$syltok != ""} {
		lappend olabs [list $sylbeg $syldur $syltok]
	    }
	}
    }
    return $olabs
}

proc MakeSylLabs {ids {labeltype "timit"} {mode "normal"} {keepdias 0} {mergecls 0} {verbose 0}} {
    # Take a slit of file IDs and create syllable label files for them
    # (using the new methods i.e. retain the diacritics)
    # $mode determines whether syls are actually written, and 
    # how verbose the output is.
    global label_framerate

    switch $mode {
	normal		{set dowrite 1; set echo 0}
	rehearse	{set dowrite 0; set echo 1}
	verbose		{set dowrite 1; set echo 1}
	default		{puts stderr "makeSyls: mode $mode unknown"; return}
    }

    OpenTsylb

    foreach id $ids {

    	set labfile [FullFileName $id "phn"]
	set sylfile [FullFileName $id "syl"]

	set WritePostArgs ""
	if {$labeltype == "xlabel"} {
	    set labdurs [ReadLabelFileXL $labfile]
	    set WriteFn WriteLabelFileXL
	    set WritePostArgs "0.0 1"
	} elseif {$labeltype == "timit"} {
	    set labdurs [ReadLabelFileTI $labfile $label_framerate]
	    set WriteFn WriteLabelFileTI
	    set WritePostArgs $label_framerate
	} elseif {$labeltype == "noway"} {
	    set labdurs [ReadLabelFileNW $labfile]
	    set WriteFn WriteLabelFileNW
	} elseif {$labeltype == "noway2"} {
	    set labdurs [ReadLabelFileNW2 $labfile]
	    set WriteFn WriteLabelFileNW2
	} elseif {$labeltype == "ctm"} {
	    set labdurs [ReadLabelFileCTM $labfile]
	    set WriteFn WriteLabelFileCTM
	} else {
	    puts stderr "Unknown labelfile type '$labeltype' not xlabel/timit/noway/noway2/ctm"
	    exit -1
	}

	set syllabs [SyllabifyLabels $labdurs $id $keepdias $mergecls]
	if {$syllabs == ""} {
	    puts "MakeSylLabs ABORTED on file $id"
	    return
	}

	if {$dowrite} {
	    eval set rc \[$WriteFn \$sylfile \$syllabs $WritePostArgs\]
	    if {$rc < 0} {
		puts stderr "Error writing syl file $sylfile"
		exit -1
	    }
	}
	if {$echo} {
	    puts "$id: [lmap $syllabs {nth 2}]"
	}
	if {$verbose} {
	    puts stderr "Done $id: read $labfile, wrote $sylfile"
	}
    }
}

proc syllify_lexicon {arrayName {keepdias 0} {mergecls 0} {syllabmrk "+"}} {
    upvar 1 $arrayName dctArray

    # prons is now an array of lists of pronunciations for each word
    # iterate over each word and replace with 
    set wordNames [array names dctArray]
    foreach word $wordNames {
	# get the current dictionary definition
	# this returns {{0.333 {ax n}} {0.333 en} {0.333 n}}
	set dctDef [GetDictDef dctArray $word]
	set newDctDef ""
	# iterate over each pronunciation, syllabifying
	foreach pron $dctDef {
	    # divide into prior and phones
	    set prior [lindex $pron 0]
	    set phones [lindex $pron 1]
	    # build up fake time marks for SyllabifyLabels
	    set phonesTimeMarks ""
	    foreach ph $phones {
		lappend phonesTimeMarks [list 0 0 $ph]
	    }
	    # do the syllabification
	    set syllab [SyllabifyLabels $phonesTimeMarks "?" $keepdias $mergecls]
	    # build a new pronunciation, inserting syllabmrk between syllables
	    set newpron ""
	    foreach syl $syllab {
		set newpron [concat $newpron [lindex $syl 2] $syllabmrk]
	    }
	    # add this new pronunciation to the new dictionary def
	    # strip off the final syllabmrk
	    lappend newDctDef [list $prior [lrange $newpron 0 [expr [llength $newpron] - 2]]]
	}
	# set the new dictionary definition to all of the new pronunciations
	SetDictDef dctArray $word $newDctDef
    }
}
