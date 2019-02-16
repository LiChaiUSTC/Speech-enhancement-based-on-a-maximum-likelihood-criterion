#
# labelfilefns.tcl
#
# Tcl scripts for manipulating label files (phone labels, word labels etc).
# Based on various STP and other procedure sets.
#
# 1997dec02 dpwe@icsi.berkeley.edu
# $Header: /n/abbott/da/drspeech/src/guitools/dpwetcl/RCS/labelfilefns.tcl,v 1.12 2001/12/03 15:17:02 dpwe Exp $

# Package header
set labelfilefns_vsn 0.1
package provide "Labelfilefns" $labelfilefns_vsn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#
# stp-readphn.tcl
#
# Functions for reading *.phn (or other xlabel label files) into Tcl
#
# dpwe@icsi.berkeley.edu  1996jun03

# Global for dummy header
set _labelfile_dummyHeader ""

proc ReadLabelFile {filename {warn 1}} {
    # Reads a file in NIST/ESPS labelled format into a 
    # list whose first element is the header lines (separated by \n's)
    # and whose second element is a list of pairs like
    # {time {rest of token line}}
    global _labelfile_dummyHeader

    if {[set file [Open $filename "r"]]==""} {
	Error "Unable to read '$filename' as label file"
	return {}
    }
    # Read each line
    set hdr {}
    set inhdr 1
    set rslt {}
#    set delpattern [format {*%c*} 127]
    set delpattern [format {*[%c%c-%c%c-%c]*} 127 1 8 10 31]
    while {![eof $file]} {
    	set line [string trim [gets $file]]
	# Escape any stray braces that will mess up the list handling
	regsub -all "\{" $line "\\\{" line
	regsub -all "\}" $line "\\\}" line
	regsub -all \\\[ $line "\\\[" line
	regsub -all \\\] $line "\\\]" line
	regsub -all "\"" $line "\\\"" line
	if {[regsub -all ";" $line "\\;" line] && !$inhdr && $warn} {
	    puts stderr "WARNING: semicolon in $filename at $line"
	}
	# Check for weird characters	
	if {[string match $delpattern $line]} {
	    puts stderr "$filename: odd chrs in '$line'"
	}
    	set c [string index $line 0]
        if {$inhdr} {
	    # gather the header lines more-or-less verbatim
	    append hdr "$line\n"
	    if {$c == "#"} {
		# End of header is marked by a single hash on its own
		set inhdr 0
	    }
	} else {
	    # Past the header - treat lines as tokens
	    if {[llength $line]>0 && $c != "#"} {
		lappend rslt [list [lindex $line 0] [lrange $line 1 e]]
	    }
	}
    }
    close $file
    # Record dummy header
    set _labelfile_dummyHeader $hdr
    return [list $hdr $rslt]
}

proc WriteLabelFile {filename labeldata} {
    # Converse of ReadLabelFile - take the two-element list of
    # the form returned by ReadLabelFile and write it out to a new 
    # file named $filename
    global _labelfile_dummyHeader

    if {[llength $labeldata] != 2} {
	puts stderr "WriteLabelFile: labeldata is not a two-element list ([llength $labeldata])"
	return 0
    } else {
	set hdr [lindex $labeldata 0]
	set nlabels 0
	if {$hdr == "*" && $_labelfile_dummyHeader != ""} {
	    # No header specified - use any previously-read one
	    set hdr $_labelfile_dummyHeader
	} elseif {$hdr == "*" || $hdr == ""} {
	    # Employ a dummy header
	    set dollar "\$"
	    set hdr "signal WriteLabelFileDummy\ntype 0\ncolor 121\ncomment ${dollar}Header: $dollar\nfont -misc-*-bold-*-*-*-15-*-*-*-*-*-*-*\nseparator \\;\nnfields 1\n#\n"
	}
	set tokpairs [lindex $labeldata 1]
	# Open the file
	set file [open $filename "w"]
	if {$file == ""} {
	    puts stderr "WriteLabelFile: couldn't write $filename"
	} else {
	    # Write out the header (has its own terminal linebreak)
	    regsub -all {\\("|{|}|[|]|;)} $hdr {\1} hdr
	    puts -nonewline $file $hdr 
	    # Write out each of the remaining lines
	    foreach pr $tokpairs {
		set time [lindex $pr 0]
		set toks [lindex $pr 1]
		# Strip the backquotes from escaped quotes etc
		regsub -all {\\("|{|}|[|]|;)} $toks {\1} toks
		# To please colorizer: "
		puts $file "    [format %.6f $time]  $toks"

		incr nlabels
	    }
	    close $file
	}
    }
    return $nlabels
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#!/usr/local/bin/tclsh
#
# stp-syldurs.tcl
#
# Functions to extract syllable durations
# from the Switchboard Transcription Project files
# (functions extracted from syllify.tcl)
#
# dpwe@icsi.berkeley.edu 1997jan15
#

proc LabelsToDurs {labs {filter 1} {reportEmpty 1}} {
    # Convert the label structure read from a file into things with durations
    # i.e. {etime colr tok} {etime colr tok} becomes
    # {btime dur tok} {btime dur tok}
    # Specifying that $filter is zero will prevent the stripping of h#s 
    # from the output
    set durlabs {}
    # Assume first label comes from start of file
    set lasttime 0
    foreach lab $labs {
	# Break out the label fields
	set time [lindex $lab 0]
	# Convert $time to Tcl-valid float repn?
	set time [expr $time + 0.0]
	set colr [lindex [lindex $lab 1] 0]
	set tok  [lrange [lindex $lab 1] 1 e]
	# Skip H#s or whatever
	set TOK [string toupper $tok]
#	if {$colr != 121} {
#	    puts stderr "** color=$colr, not 121"
#	}
	# Don't ignore empty labels for now
        if {!$filter || \
		($TOK != {} && $TOK != "SIL" && $TOK != "H#" \
		 && $TOK != "UNL" && $TOK != "SP")} {
	    lappend durlabs [list $lasttime [expr $time-$lasttime] $tok]
	} elseif {$TOK == {} && $reportEmpty} {
	    puts "**Empty TOK in $labs"
	}
	set lasttime $time
    }
    return $durlabs
}

proc DursToLabels {dlabs {maxdur 0.0}} {
    # Convert the {btime dur tok} lists to {etime colr tok} lists
    # If necessary, append an h# to place the last label at $maxdur
    set olabs ""
    # Color tag for all outputs
    set colr 121
    # Token for gaps
    set gaptoken "h#"
    set ltime 0.0
    # tolerate gaps up to 5 ms
    set tol 0.005
    foreach lab $dlabs {
	lassign $lab time dur tok
	if {$time < ($ltime-$tol)} {
	    puts stderr "DursToLabs: overlap at $lab"
	}
	if {$time > ($ltime + $tol)} {
	    # Gap - need to insert an H#
	    lappend olabs [list $time [concat $colr $gaptoken]]
	}
	# Add the new label
	set ltime [expr $time+$dur]
	lappend olabs [list $ltime [concat $colr $tok]]
    }
    # Append a final one if necessary
    if {$ltime < ($maxdur - $tol)} {
	lappend olabs [list $maxdur [concat $colr $gaptoken]]
    }
    return $olabs
}

# New versions of read/write label files work in dur domain
proc ReadLabelFileXL {filename} {
    # Read an ? xlabel file $filename, and return its contents in 
    # "durs" format i.e. {btime dur tok} {btime dur tok} ...
    set labs [LabelsToDurs [lindex [ReadLabelFile $filename] 1] 0 0]
    return $labs
}

proc WriteLabelFileXL {filename durlabs {maxdur 0.0} {reuseRead 0}} {
    # Write an ? xlabel file $filename to contain the dur-format
    # labels in $durlabels.   Optional $maxdur ensures a final h# label.
    # If $reuseRead, indicate that the header should be the last one 
    # read by ReadLabelFile.
    set labs [DursToLabels $durlabs $maxdur]
    if {$reuseRead} {
	# flag to WriteLabelFile to reuse the last read header
	set hdr "*"
    } else {
	set hdr ""
    }
    return [WriteLabelFile $filename [list $hdr $labs]]
}

proc ReadLabelFileNW {filename {framestp ""} {forcezero 1}} {
    # Read a noway-format decodefile $filename, and return its contents in 
    # "durs" format i.e. {btime dur tok} {btime dur tok} ...
    # If $forcezero, the times are all shifted so that the first start 
    # time corresponds to zero (to get around cumulative times reported 
    # in a single noway session).
    global frame_step noway_header
    if {$framestp == ""} {
	set framestp $frame_step
    }
    # Make sure frameftp makes sense as *seconds* (def as milliseconds)
    if {$framestp > 0.5} {
	set framestp [expr $framestp/1000.0]
    }
    if {$forcezero} {
	set zerotime ""
    } else {
	set zerotime 0.0
    }
    if {[set inf [Open $filename "r"]] == ""} {
	puts stderr "ReadLabelFileNW: couldn't open $filename"
	return ""
    }
    set labs ""
    set first 1
    while {![eof $inf]} {
	set line [string trim [gets $inf]]
	if {![eof $inf]} {
	    if {$first} {
		# First line is result
		set noway_header $line
		set first 0
	    } else {
		if {$line != ""} {
		    # e.g "122 132  iy7 (128)"
		    lassign $line startframe endframe phone stuff
		    set starttime [expr $startframe*$framestp]
		    set endtime   [expr $endframe*$framestp]
		    # remove phone count
		    regsub {[0-9]+} $phone "" phone
		    # map phones
		    regsub interword-pause $phone "h#" phone
		    # build output
		    set dur [expr $endtime - $starttime]
		    if {$zerotime == ""} {
			set zerotime $starttime
		    }
		    lappend labs [list [expr $starttime-$zerotime] $dur $phone]
		}
	    }
	}
    }
    close $inf
    return $labs
}

proc ReadLabelFileNW2 {filename {forcezero 1}} {
    # Read a *NEW* noway-format decodefile $filename, e.g. 
    #   0.380 0.040 ao4 (21)
    # and return its contents 
    # in "durs" format i.e. {btime dur tok} {btime dur tok} ...
    # If $forcezero, the times are all shifted so that the first start 
    # time corresponds to zero (to get around cumulative times reported 
    # in a single noway session).
    global frame_step noway_header
    if {$forcezero} {
	set zerotime ""
    } else {
	set zerotime 0.0
    }
    if {[set inf [Open $filename "r"]] == ""} {
	puts stderr "ReadLabelFileNW2: couldn't open $filename"
	return ""
    }
    set labs ""
    set first 1
    while {![eof $inf]} {
	set line [string trim [gets $inf]]
	if {![eof $inf]} {
	    if {$first} {
		# First line is result
		set noway_header $line
		set first 0
	    } else {
		if {$line != "" && [string index $line 0] != "#"} {
		    # e.g "0.380 0.040 ao4 (21)"
		    lassign $line starttime dur phone stuff
		    # remove phone count
		    regsub {[0-9]+} $phone "" phone
		    # map phones
		    regsub "interword-pause" $phone "h#" phone
		    # build output
		    if {$zerotime == ""} {
			set zerotime $starttime
		    }
		    lappend labs [list [expr $starttime-$zerotime] $dur $phone]
		}
	    }
	}
    }
    close $inf
    return $labs
}

proc WriteLabelFileNW {filename labels {maxdur 0.0} {framestp ""} {nowayhdr ""}} {
    # Write a noway-format phondecode file $filename to contain the dur-format
    # labels in $labels.   Optional $maxdur ensures a final h# label.
    global frame_step noway_header
    if {$framestp == ""} {
	set framestp $frame_step
    }
    # Make sure frameftp makes sense as *seconds* (def as milliseconds)
    if {$framestp > 0.5} {
	set framestp [expr $framestp/1000.0]
    }
    # maybe use the global header
    if {$nowayhdr == "" && [info exists noway_header]} {
	set nowayhdr $noway_header
    }
    if {$nowayhdr == ""} {
	set nowayhdr "1 DUMMY"
    }
    # Write the output file
    if {[set file [Open $filename "w"]] == ""} {
	puts stderr "WriteLabelFileNW: couldn't open $filename"
	return -1
    }
    # Write the header
    puts $file $nowayhdr
    # Write the labels
    set nlabels 0
    foreach triple $labels {
	lassign $triple start dur tok
	# Map the labels
	if {$tok == "h#"} {
	    set tok "interword-pause"
	    set count ""
	} else {
	    # Fake the extra info
	    set count 0
	}
	set score 0
	puts $file "[expr round($start/$framestp)] [expr round(($start+$dur)/$framestp)] ${tok}${count} ($score)"
	incr nlabels
    }
    # Blank line at end
    puts $file ""
    close $file
    return $nlabels
}

proc ReadLabelFileCTM {filename {rq_uttid ""} {forcezero 1}} {
    # CTM files are written by noway to record the alignment and score 
    # of *words* in an utterance.  I'd like to read & display this 
    # timing information, so read them in....
    # Each row is like:
    #      uttid chan start dur                 token  AMscore
    # i.e.
    #      1     A    0.000 0.304               <SIL>  -4.9232
    # If $rq_uttid is not blank, scan through the file to 
    # find the matching uttid.  Otherwise, ignore uttids.
    # If $forcezero, the times are all shifted to make the first starttime 
    # zero (to get around cumulative times reported 
    # in a single noway session).
    if {$forcezero} {
	set zerotime ""
    } else {
	set zerotime 0.0
    }
    if {[set inf [Open $filename "r"]] == ""} {
	puts stderr "ReadLabelFileCTM: couldn't open $filename"
	return ""
    }
    set labs ""
    while {![eof $inf]} {
	set line [string trim [gets $inf]]
	if {![eof $inf]} {
	    if {$line != ""} {
		# e.g "1 A 0.000 0.304               <SIL>  -4.9232"
		lassign $line uttid chan start dur tok amscore
		if {$rq_uttid == "" || $uttid == $rq_uttid} {
		    # map tokens?
		    #regsub <SIL> $tok "h#" tok
		    # build output
		    if {$zerotime == ""} {
			set zerotime $start
		    }
		    lappend labs [list [expr $start-$zerotime] $dur $tok]
		}
	    }
	}
    }
    close $inf
    return $labs
}

# Similar functions for timit-format label files
proc ReadLabelFileTI {filename {srate ""}} {
    # Read a timit label file $filename and return its contents 
    # in "durs" format - {btime dur tok}
    # Overlaps are ignored
    global samplerate
    if {$srate == ""} {
	set srate $samplerate
    }    
    set labs ""
    set sampdur [expr 1.0/$srate]
    if {[set file [Open $filename "r"]] == ""} {
	puts stderr "ReadLabelFileTI: couldn't open $filename"
	return ""
    }
    set done 0
    set first 1
    while {!$done} {
	set line [gets $file]
	if {![eof $file]} {
	    if {[string length [string trim $line]] > 0 \
		    && [string index $line 0] != "#"} {
		lassign $line thisstartsamp thisendsamp thistok
		if {!$first} {
		    # Do you want the boundaries to match exactly, or
		    # do we preserve gaps and overlaps in original data?
		    # (1) force previous end to be this start
		    #set end $thisstartsamp
		    # (2) Use the overlapping ends from the actual file
		    set end $lastendsamp
		    set dur [expr ($end-$laststartsamp)*$sampdur]
		    set start [expr $laststartsamp*$sampdur]
		    lappend labs [list $start $dur $lasttok]
		} else {
		    set first 0
		}
		set laststartsamp $thisstartsamp
		set lastendsamp $thisendsamp
		set lasttok $thistok
	    }
	} else {
	    set done 1
	}
    }
    # Reached EOF - write out last sample
    if {!$first} {
	set start [expr $laststartsamp*$sampdur]
	set dur   [expr ($lastendsamp - $laststartsamp)*$sampdur]
	lappend labs [list $start $dur $lasttok]
    }
    close $file
    return $labs
}

proc WriteLabelFileTI {filename labels {srate ""}} {
    # Write a TIMIT-format label file from a {start dur tok} list
    global samplerate
    if {$srate == ""} {
	set srate $samplerate
    }
    set srate [expr 0.0+$srate]
    if {[set file [Open $filename "w"]] == ""} {
	puts stderr "WriteLabelFileTI: couldn't open $filename"
	return -1
    }
    set nlabels 0
    foreach triple $labels {
	lassign $triple start dur tok
	puts $file "[expr round($srate*$start)] [expr round($srate*($start+$dur))] $tok"
	incr nlabels
    }
    close $file
    return $nlabels
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# CanvLabels.tcl
#
# Draw xlabel-style file on a canvas
#
# after tksound/tsglabel.tcl and cgi-bin/sgramImg.cgi
#
# 1997apr12 dpwe@icsi.berkeley.edu

## To get the label file access routines
##set stptcldir "/u/dpwe/projects/stp/scripts"
#set stptcldir "/u/stp/share/lib/stp-tcl"
#source "$stptcldir/stp-readphn.tcl"
#source "$stptcldir/stp-syldurs.tcl"

proc CanvAddLabels {canv labs xbase ybase width height timestart timeend {sense 1} {tags labels} {font ""}} {
    # Add xlabels-style labels on a canvas
    # if $sense, labels are 'right-aligned' a la xlabel
    # A good font (for screen & printing) is: "*helvetica*medium-r*--10*"
    global canvasFont _labels_bindcanvas _labels_tag
    if {$font == ""} {
	set font $canvasFont
    }
    set timescale [expr $width/($timeend-$timestart)]    
    # Put a black line at the top
    $canv create line $xbase $ybase [expr $xbase+$width] $ybase -tags $tags
    set xfix 0
    set yfix -1
    set lastx 9999
    set lasty 0
    set lasttok "*dummy*"
    if {!$sense} {
    	set labs [reverse $labs]
    } else {
	# Add dummy last tok to make sure it's displayed
	lassign [lindex $labs e] time dur tok
	lappend labs [list [expr $time+$dur] 0.0 ""]
    }
    foreach lab $labs {
	lassign $lab time dummy tok
	# Be sure to escape any square brackets in $tok
	# (in a pattern, backslash-bracket means bracket, since it's 
	#  a special chr.  But in a replacement, a bracket *isn't* a 
	#  specical chr, so backslash-bracket is treated literally; 
	#  two characters are inserted as the replacement).
	regsub -all {\[} $tok {\[} tok
	regsub -all {\]} $tok {\]} tok
	set x [expr $xbase + int(($time - $timestart)*$timescale)]
	set y $ybase
	if {$x >= $xbase && $x < ($xbase+$width)} {
	    if {$sense} {
		# Use eval to undo escaping of quotes etc in labels
		eval "set thistok \"$lasttok\""
	    } else {
		eval "set thistok \"$tok\""
	    }
	    lassign [CanvTextDims $canv $thistok $font] wdth hght
	    if {$sense} {
		# set x_tx [expr $x+1-$wdth-$xfix]
		set x_tx [expr $x+2-$wdth]
		set right [expr $x+2-$wdth]
		set anch "sw"
	    } else {		
		set x_tx [expr $x+2]
		set right [expr $x+$wdth-2]
		set anch "sw"
	    }
	    if {($wdth+$xfix) >= abs($x-$lastx)} {
		# Overlap with preceding label (right to left) - move down
		# (if it would fit)
		if {[expr $lasty+2*$hght] <= [expr $y+$height]} {
		    set y [expr $lasty + $hght]
		}
	    }
	    if {$thistok != "*dummy*"} {
		set textbottom [expr $y+$hght]
		$canv create text $x_tx $textbottom -text $thistok -anchor $anch \
		    -font $font -tags $tags
		# Add lines
		set bottom [expr $y+$hght+$yfix]
		$canv create line $x $ybase $x $bottom $right $bottom \
		    -width 0.4 -tags $tags
		# I think -width is cast to an int - so 0.5 == 1, hairline == 0 
	    }
	    # Remember position of this label
	    set lastx $x
	    set lasty $y
	}
	set lasttok $tok
    }
    # Permit moving (after CanvasItem)
    if {[catch {set _labels_bindcanvas [$canv childsite]}]} {
	# childsite rejected - must be a plain canvas widget
	set _labels_bindcanvas $canv
    }
    set _labels_tag $tags
    $_labels_bindcanvas bind $_labels_tag <1> "LabelsClick %x %y"
}

proc LabelsClick {mx my} {
    global _labels_basex _labels_basey _labels_bindcanvas
    set _labels_basex $mx
    set _labels_basey $my
    bind $_labels_bindcanvas <B1-Motion> "LabelsMove %x %y"
    bind $_labels_bindcanvas <ButtonRelease-1> "LabelsRelease %x %y"
}
proc LabelsMove {mx my} {
    global _labels_basex _labels_basey _labels_bindcanvas _labels_tag
    set dx [expr $mx-$_labels_basex]
    set dy [expr $my-$_labels_basey]
    # No X moves
    set dx 0
    incr _labels_basex $dx
    incr _labels_basey $dy
    $_labels_bindcanvas move $_labels_tag $dx $dy
}
proc LabelsRelease {mx my} {
    global _labels_bindcanvas
    bind $_labels_bindcanvas <B1-Motion> ""
    bind $_labels_bindcanvas <ButtonRelease-1> ""
}

# 1998apr25: not exactly labelfilefn, but used in conjunction 
# (in xlabel2pfile)
# - - - - - - /u/stp/src/stp-tcl/stp-statusfile-fns.tcl - - - - - -
#!/usr/local/bin/tclsh
#
# stp-statusfile-fns.tcl
#
# Script to grab newly-approved files from Candice's screening 
# and add them to the pool of allocatable files for the other 
# transcribers.
#

# [...]

proc WavSamplesSafe {wavfile} {
    # Return the count of sample frames in the named soundfile, 
    # attempting to get the 'true' value from the header
    if {[file exist $wavfile]} {
	set cmdargs ""
	if {[string match "*.sd" $wavfile]} {
	    set cmdargs "-S ESPS"
	}
        catch {eval exec sndcat -q -v $cmdargs $wavfile} cmdrslt
	if {[regexp "data bytes *(\[0-9\]+)" $cmdrslt all nsamps] == 0} {
	    Warn "WavSamplesSafe: no nsamps from sndcat for $wavfile ($cmdrslt)"
	    return 0
	}
	# sloppy - what if song is called "16 bit Stereo" ?
	# but actual spaces before and after are real
	if {[string match "* 16 bit *" $cmdrslt]} {set nsamps [expr $nsamps/2]}
	if {[string match "* 32 bit *" $cmdrslt]} {set nsamps [expr $nsamps/4]}
	if {[string match "* Stereo *" $cmdrslt]} {set nsamps [expr $nsamps/2]}
	if {[regexp {([0-9]+) channels} $cmdrslt all chans]} {set nsamps [expr $nsamps/$chans]}
    } else {
	set nsamps 0
    }
    return $nsamps
}

proc WavSamplesQuick {file} {
    # Return the duration in samples of the specified file
    # Sadly, we have to hack this to work with *.sd files
    set nsamps 0
    if {[file exist $file]} {
	set samplesize 2
	set filebytes [file size $file]
	if {[string match "*.sd" $file]} {
	    # Empirically, sd files have 631 (635?) -byte headers
	    #set headersize 635
	    set headersize 631
	    set nsamps [expr ($filebytes-$headersize)/$samplesize]
	} elseif {[string match "*.wav" $file]} {
	    # Assume nist
	    set nsamps [expr (($filebytes-1024)/$samplesize)]
	} else {
	    puts stderr "WaveSamplesQuick: don't know how to treat $file"
	}
    }
    return $nsamps
}

# [...]

# Addition, 1998nov01 - was in syllify.tcl, then syllifyfns.tcl

proc FullFileName {id type} {
    # Construct a full file name of type $type from the id $id
    # using either ${type}filecmd or ${type}directory and ${type}extension
    # globals
    upvar \#0 ${type}filecmd filecmd
    upvar \#0 ${type}directory directory
    upvar \#0 ${type}extension extension
    
    if {$filecmd != ""} {
    	if {![regsub -all "%u" $filecmd $id fullcmd]} {
    	    set fullcmd "$filecmd$id"
	}
	set filename [eval "exec $fullcmd"]
    } else {
    	set filename "$directory/${id}$extension"
    }
    return $filename
}

# Addition 1999may24 to read a segment of a pflab file as labdurs

proc ReadLabelFilePF {filename uttno {framestp ""} {ipformat "pfile"}} {
    # Read a single utterance from an ennumerated-label file (such as a pflab)
    # but return it in labdur format.
    # ** YOU MUST HAVE ALREADY SET UP A PHONESET FILE WITH ReadPhoneset
    global frame_step noway_header
    # Setup the framestp
    if {$framestp == ""} {
	set framestp $frame_step
    }
    # Make sure framestp makes sense as *seconds* (def as milliseconds)
    if {$framestp > 0.5} {
	set framestp [expr $framestp/1000.0]
    }
    # Open the pipe to stream in the labels
    set f [open "| labcat -opf ascii -sr $uttno $filename -ipf $ipformat" "r"]
    set lastlabid ""
    set rslt {}
    set time 0.0
    while {![eof $f]} {
	set d [string trim [gets $f]]
	if {$d != ""} {
	    # each line is "utt# frm# labcode"
	    set labid [lindex $d 2]
	    if {$labid != $lastlabid} {
		if {$lastlabid != ""} {
		    # flush last label
		    set dur [expr $time - $start]
		    set tok [PhoneName $lastlabid]
		    lappend rslt [list $start $dur $tok]
		}
		# Setup for next label
		set start $time
		set lastlabid $labid
	    }
	    set time [expr $time + $framestp]
	}
    }
    # Append last label
    if {$lastlabid != ""} {
	set dur [expr $time - $start]
	if {$dur > 0.0} {
	    set tok [PhoneName $labid]
	    lappend rslt [list $start $dur $tok]
	}
    }
    # Clean up & return
    close $f
    return $rslt
}

