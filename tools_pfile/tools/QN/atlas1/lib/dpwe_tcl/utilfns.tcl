#
# utilfns.tcl
#
# A bunch of little utility subroutines for tcl
#
# dpwe@media.mit.edu 1995aug06
# $Header: /u/drspeech/src/guitools/dpwetcl/RCS/utilfns.tcl,v 1.25 1999/01/21 23:16:35 dpwe Exp $
#

# Package header
set dpwe_utilfns_vsn 0.1
package provide "Dpwe_Utilfns" $dpwe_utilfns_vsn

# Avoid having to call /u/drspeech/share/bin/speech_arch the whole time
proc SpeechArch {} {
    # Figure out the appropriate keyword for this speech arch
    # (equivalent to shell program /u/drspeech/share/bin/speech_arch)
    # (algorithm copied from drspeech.pm)
    set mach [exec uname -m]
    set os   [exec uname -r]
    set arch "unknown"
    set pat "${mach}:${os}"
    if {[regexp {^sun4.*:5\.} $pat]}	{set arch "sun4-sunos5"}
    if {[regexp {^sun4.*:4\.} $pat]}	{set arch "sun4"}
    if {[regexp {^IP.*:4\.} $pat]}	{set arch "iris"}
    if {[regexp {^IP.*:5\.} $pat]}	{set arch "iris-irix5"}
    if {[regexp {^i.86:.*} $pat]}	{set arch "i586-linux"}
    if {$arch == "unknown"} {
	error "SpeechArch: unknown architecture ('$pat')"
    }
    return $arch
}

# Added (from ~/projects/NUMBERS95/run-tests) 1997oct27
proc SubsFile {templatename outfile subs} {
    # Build a file from a template.  The file $templatename will 
    # be copied to the open stream $outfile, except for substitutions:  
    # $subs is a list of pairs; the first is a token, which 
    # will be replaced by the second (the value) in writing 
    # the file.
    if {[set infile [Open $templatename "r"]] == ""} {
	Warn "SubsFile: unable to read template '$templatename'"
	return -1
    }
    while {![eof $infile]} {
	set line [gets $infile]
	if {![eof $infile]} {
	    # Perform the substitutions
	    foreach pair $subs {
		lassign $pair patt repl
		regsub -all $patt $line $repl line
	    }
	    # Write to output
	    puts $outfile $line
	}
    }
    # Don't close the output file since we didn't open it
    close $infile
}

proc GetICSIargs {{acceptlist ""}} {
    # Process argc & argv in the manner of /u/drspeech/share/lib/icsiargs.tcl
    # If acceptlist is nonempty, accept only args in that list
    # Format is param=val, with no spaces around the =
    # Return list of params that were set.
    global argv argc
    set outargv ""
    set outargc 0
    set gotvars {}
    foreach p $argv {
	if {[string match "*=*" $p]} {
	    regexp {([^=]*)=(.*)} $p all param val
	    if {$acceptlist != ""} {
		if {[lsearch -exact $acceptlist $param] < 0} {
		    puts stderr "GetICSIargs: arg \"$param\" is not one of: $acceptlist"
		    # Maybe exit? or return null?  Just continue for now
		    continue
		}
	    }
	    uplevel #0 "set $param \{$val\}"
	    lappend gotvars $param
	} else {
	    # param doesn't contain "="
	    lappend outargv $p
	    incr outargc
	}
    }
    set argv $outargv
    set argc $outargc
    return $gotvars
}

# More modern replacements for GetICSIargs..
# Use like:
# set argdef { \
#    {"" "tag for this decode" "NOTAG"} \
#    {"wts" "weights file to use" ""} \
#    {"ftrs" "path component defining features" "msg1N"} \
#    {"nftrs" "width of feature vec (from pfile)" "0"} \
#    {"hus"   "number of hidden units" "2000"} \
#    {"template" "template for bnr.params" "single"} \
#    {"decodemachine" "machine for decodes" "gin"} \
#}
#
#set argv_out [DrspeechGetargs $argdef $argv]
#
#if {$wts == "" || $argv_out != ""} {
#    puts stderr "wts='$wts' is empty, or argv='$argv_out' is not empty"
#    DrspeechUsage $argdef
#    exit -1
#}


proc DrspeechGetargs {defs argv} {
    # Get a bunch of args from the command line args $argv
    foreach def $defs {
	lassign $def name desc dflt
	if {[set ix [lsearch -glob $argv "${name}=*"]] == -1} {
	    uplevel 1 "set $name \{$dflt\}"
	} else {
	    regexp "=(.*)" [lindex $argv $ix] all val
	    uplevel 1 "set $name \{$val\}"
	    # remove used $argv element
	    set argv [lreplace $argv $ix $ix]
	}
    }
    return $argv
}

proc DrspeechUsage {defs} {
    # Generate the usage message
    global argv0
    puts stderr "Options for $argv0:"
    foreach def $defs {
	lassign $def name desc dflt
	puts "  $name=<val>   $desc ($dflt)"
    }
}

proc quit {} {
    exit 
}

set _exit_on_error 1

proc Error {msg} {
    # Generic error handler
    global argv0 _exit_on_error
    puts stderr "$argv0: Error: $msg"
    if {$_exit_on_error} {
	exit -1
    }
    #error "$argv0: Error: $msg"
}

proc Warn {msg} {
    # Generic error handler
    global argv0
    puts stderr "[file tail $argv0]: Warn: $msg"
}

proc Assert {test} {
    # assertion tester.  Evaluates $test, bombs if not true
    set rslt [uplevel 1 "expr $test"]
    if {!$rslt} {
    	Error "Assertion failed: $test"
    	error "bad assert"
    }
}

proc Open {filename {mode "r"}} {
    # Like open, but returns failures as null tokens rather than 
    # crashing out
    if {[catch {set file [open $filename $mode]} err]} {
	return ""
    } else {
	return $file
    }
}

proc Incr {varname {amt 1}} {
    # increment a variable; create it if it doesn't yet exist
    uplevel 1 "if {\[info exists $varname\]==0} {set $varname 0}"
    uplevel 1 "incr $varname $amt"
}

proc LoadProfileFile {name} {
    # Load a file called $HOME/$name which defines a set of 
    # stp variables.  Return the names of the variables that were set
    global env
    set fn "$env(HOME)/$name"
    if {[file readable $fn]} {
	uplevel #0 "source $fn"
    }
    return ""
}

proc SaveProfileFile {name vars} {
    # Write a status file called $HOME/$name which holds the values 
    # of the variables named in $vars
    global env
    set fn "$env(HOME)/$name"
    set body ""
    if {[set f [Open $fn "r"]] != ""} {
	set body [lwithout [split [read $f] "\n"] ""]
	close $f
    }
    foreach v $vars {
	set setline "set $v \{[uplevel 1 "set $v"]\}"
	if {[set ix [lsearch -glob $body "set $v *"]] != -1} {
	    # Already specify this variable
	    set body [lreplace $body $ix $ix $setline]
	} else {
	    # Not in file yet
	    lappend body $setline
	}
    }
    # Write the file
    if {[set f [Open $fn "w"]] != ""} {
	puts $f [join $body "\n"]
	close $f
    } else {
	puts stderr "SaveStatusFile: couldn't write '$fn'"
    }
}

proc WinWidth {win} {
    # figure the width of a window's interior (work area) by asking 
    # the window manager, then knocking off the borderwidth
    set bw [$win cget -borderwidth]
    set w  [winfo width $win]
    # knock off two lots of $bw, one for each side
    return [expr "$w - 2*$bw"]
}

proc WinHeight {win} {
    # figure the height of a window's interior (work area) by asking 
    # the window manager, then knocking off the borderwidth
    set bw [$win cget -borderwidth]
    set h  [winfo height $win]
    # knock off two lots of $bw, one for each side
    return [expr "$h - 2*$bw"]
}

proc CanvTextDims {cnv {text "N"} {font {}}} {
    # return the w and h for the given text & font if drawn in the given 
    # canvas.  If omitted, font uses default canvas font; 
    # With no text specified, returns the size of an upper-case N.
    if { $font == "" }  {
	# no font specified, allow default to happen
	set tag [$cnv create text 10 10 -text $text]
    } else {
	set tag [$cnv create text 10 10 -text $text -font $font]
    }
    set bbx [$cnv bbox $tag]
    $cnv delete $tag
    return [list [expr "[lindex $bbx 2]-[lindex $bbx 0]"] \
	    [expr "[lindex $bbx 3]-[lindex $bbx 1]"]]
}

proc max args {
    # Treats args as a list of numbers, returns the largest
    set list [eval "concat $args"]
    set rslt [lindex $list 0]
    foreach el $list {
	if { $el > $rslt } {
	    set rslt $el
	}
    }
    return $rslt
}

proc min args {
    # Treats args as a list of numbers, returns the smallest
    set list [eval "concat $args"]
    set rslt [lindex $list 0]
    foreach el $list {
	if { $el < $rslt } {
	    set rslt $el
	}
    }
    return $rslt
}

proc sum args {
    # Treats args as a list of reals, returns the sum
    set list [eval "concat $args"]
    # A decimal point in the list means treat them as reals
    if {[string first "." $list] >= 0} {
	set rslt 0.0
    } else {
	set rslt 0
    }
    foreach el $list {
	set rslt [expr $rslt+$el]
    }
    return $rslt
}

proc lwithout {args} {
    # usage: lwithout ?-exact|-glob? list val
    # Remove all instances of $val from $list; return the remainder
    lassign [processProcArgs $args] match list val
    if {$match == ""} {set match -exact}
    while { [set ix [lsearch $match $list $val]] >= 0 } {
        # null out the $ix'th value
        set list [lreplace $list $ix $ix]
    }
    return $list
}

proc lwithoutl {args} {
    # usage: lwithout ?-exact|-glob? list stoplist
    # Remove all instances of els of $stoplist from $list; return the remainder
#    set outlist {}
#    foreach el $list {
#	if {[lsearch $match $stoplist $el]==-1} {
#	    lappend outlist $el
#	}
#    }
    lassign [processProcArgs $args] match list stoplist
    if {$match == ""} {set match -exact}
    if {$match == "-exact"} {
	return [lwithoutl_simple $list $stoplist]
    } else {
	set outlist $list
	foreach stop $stoplist {
	    set outlist [lwithout $match $outlist $stop]
	}
	return $outlist
    }
}

proc lwithoutl_simple {list1 list2} {
    # Return a list of the elements from list1 that aren't in list2
    # New, 10x faster version uses array
    # Make a tmp array
    foreach el $list2 {
	set tmpArray($el) 1
    }
    set rslt {}
    foreach el $list1 {
	if {[array get tmpArray $el] == ""} {
	    lappend rslt $el
	}
    }
    return $rslt
}

proc lintersect {list1 list2} {
    # Return a list of the elements in both lists
    # Faster if $list1 is the longer of the pair
    # Make a tmp array
    foreach el $list1 {
	set tmpArray($el) 1
    }
    set isect {}
    foreach el $list2 {
	if {[array get tmpArray $el] != ""} {
	    lappend isect $el
	    # Don't report it again
	    unset tmpArray($el)
	}
    }
    return $isect
}

proc lunion {list1 list2} {
    # Return a list of the elements in either list; each element 
    # reported only once.
# fast but weird output order
#    foreach el [concat $list1 $list2] {
#	set tmpArray($el) 1
#    }
#    return [array names tmpArray]
    set union ""
    foreach el [concat $list1 $list2] {
	if {[array get tmpArray $el] == ""} {
	    set tmpArray($el) 1
	    lappend union $el
	}
    }
    return $union
}

proc lmap {list fn} {
    # Return a new list where each element is the result of applying 
    # $fn to an element of $list - but those elements form a single arg
    set out {}
    foreach el $list {
	lappend out [eval "$fn {$el}"]
    }
    return $out
}

proc nth {n args} {
    # Quick function to return just its nth argument.
    # List is the last arg, thus can be used as a prefix fn 
    # for lmap
    # lists likely to have been split into args by eval
    set list [join $args " "]
    return [lindex $list $n]
}

proc cons {head rest} {
    # Return a list whose car is $head and whose cdr is $rest
    set out [concat [list $head] $rest]
}

proc reverse list {
    # Return the list with its top-level elements in reverse order
    set out {}
    foreach e $list {
	set out [cons $e $out]
    }
    return $out
}

proc resetArray {args} {
    # from original itcl examples
    foreach name $args {
	uplevel 1 "catch \"unset $name\""
	uplevel 1 "set ${name}(0) \"make-this-an-array\""
	uplevel 1 "unset ${name}(0)"
    }
}

proc getMeanSd {data {ignorevals {{}}}} {
    # $data is a list of numbers; calculate their mean and sd
    # return [list $mean $sd $n]
    # Values from $ignorvals will not be included
    set sum 0.0
    set sumsq 0.0
    set n 0
    foreach x $data {
	# maybe reject certain elements
	if {[lsearch -exact $ignorevals $x]==-1} {
	    set sum   [expr $sum+$x]
	    set sumsq [expr $sumsq + ($x*$x)]
	    incr n
	}
    }
    if {$n > 0} {
	set mean [expr $sum/$n]
	# rounding problems with tcl
	set sqmean [expr $mean*$mean]
	set meansq [expr $sumsq/$n]
	if {$sqmean >= $meansq} {
	    set sd 0.0
	} else {
	    set sd [expr sqrt($meansq - $sqmean)]
	}
    } else {
	set sd 0.0
	set mean 0.0
    }
    return [list $mean $sd $n]
}

proc lassign {list args} {
    # take a list and some number of variable names; set the 
    # variables to the corresponding elements of the list
    set i 0
    foreach arg $args {
	set v [lindex $list $i]
	#regsub -all {([^\])([{}])} $v {\1\\\2} v
	#uplevel 1 "set $arg \{$v\}"
	uplevel 1 "set $arg [list $v]"
	incr i
    }
}


proc car {list} {
    # lisp's car - first element of a list
    return [lindex $list 0]
}

proc cdr {list} {
    # lisp's cdr - list without first element
    return [lrange $list 1 e]
}

proc last {list} {
    # return the last top-level element of a list
    return [lindex $list [expr [llength $list]-1]]
}

proc uniqueCommand {stem {startFrom 0}} {
    # append increasing integers to $stem until we find one that isn't
    # already a command.  Integers start search from $startFrom
    set existingCmds [info commands ${stem}*]
    set i $startFrom
    set found 0
    while {!$found} {
	set cmdName ${stem}$i
	if {[lsearch -exact $existingCmds $cmdName]==-1} {
	    set found 1
	}
	incr i
    }
    return $cmdName
}

proc getenv {envvar {dflt ""}} {
    # Return the environment variable, or $dflt if it doesn't exist
    global env
    if {[info exists env($envvar)]} {
	return $env($envvar)
    } else {
	return $dflt
    }
}

proc processProcArgs {arglist} {
    # Return the list $args parsed into a sublist of "-"-prefixed switches 
    # and remaining regular args
    set switches ""
    set otherargs ""
    set moreswitches 1
    set nargs [llength $arglist]
    for {set i 0} {$i<$nargs} {incr i} {
	if {$moreswitches && [string index [lindex $arglist $i] 0] == "-"} {
	    lappend switches [lindex $arglist $i]
	} else {
	    lappend otherargs [lindex $arglist $i]
	    # prevent subsequent arg elements from being taken as switches
	    set moreswitches 0
	}
    }
    return [cons $switches $otherargs]
}

proc regmod {args} {
    # args are [-switches] pattern string replacement
    # Perform regsub and return the result
    # Separate switches from other args
    lassign [processProcArgs $args] switches pattern string repl
    # Result will be just the string if regsub fails
    set rslt $string
    eval regsub $switches \{$pattern\} \{$string\} \{$repl\} rslt
    return $rslt
}

proc stringreplace {string first last replacement} {
    # Like list replace, replace a certain range of chrs in a string
    return "[string range $string 0 [expr $first-1]]$replacement[string range $string [expr $last+1] e]"
}

proc uptochr {chr str} {
    # Return initial portion of $str up until the first $chr (or all of it)
    set x [string first $chr $str]
    if {$x >= 0} {
	set str [string range $str 0 [expr $x-1]]
    }
    return $str
}

proc GetCLVars {argv} {
    # Take a series of command line args of form var=val
    # and set the corresponding Tcl global variables
    # Return the names of the set variables.
    set varnames {}
    foreach m $argv {
	set varconts [split $m "="]
	if {[llength $varconts]==2} {
	    set varname [lindex $varconts 0]
	    global $varname
	    set $varname [lindex $varconts 1]
	    lappend varnames $varname
	} else {
	    puts stderr "GetCLVars: couldn't parse '$m' as 'var=val'"
	}
    }
    return $varnames
}

proc _compareCarNum {a b} {
    # For sorting {count token} lists, treat inputs as lists whose 
    # first element is numeric, and compare just those first 
    # elements numerically
    set n1 [lindex $a 0]
    set n2 [lindex $b 0]
    if {$n1 < $n2} {
	return -1
    } elseif {$n1 > $n2} {
	return 1
    } else {
	return 0
    }
}

proc UniqFreq {list} {
    # Take a list of tokens and return a list of the unique ones 
    # alongside their frequencies of occurrence.  Thus 
    # "a b a c c a d" gives "{3 a} {2 c} {1 b} {1 d}"
    set slist [lsort $list]
    set op ""
    set lastel [lindex $slist 0]
    set count 1
    foreach el [lrange $slist 1 e] {
	if {$el != $lastel} {
	    lappend op [list $count $lastel]
	    set count 0
	}
	set lastel $el
	incr count
    }
    lappend op [list $count $lastel]
    return [lsort -command _compareCarNum -decreasing $op]
}

if {[info var __test__] != ""} {
#
# test-utils.tcl
#
# Test code for the dpweutilfns
#
# 1996oct23 dpwe@icsi.berkeley.edu
# $Header: /u/drspeech/src/guitools/dpwetcl/RCS/utilfns.tcl,v 1.25 1999/01/21 23:16:35 dpwe Exp $

proc ResultCode code {
    switch $code {
        0 {return TCL_OK}
        1 {return TCL_ERROR}
        2 {return TCL_RETURN}
        3 {return TCL_BREAK}
        4 {return TCL_CONTINUE}
    }
    return "Invalid result code $code"
}

proc OutputTestError {id command expectCode expectResult resultCode result} {
    puts stderr "======== Test $id failed ========"
    puts stderr $command
    puts stderr "==== Result was: [ResultCode $resultCode]:"
    puts stderr $result
    puts stderr "==== Expected : [ResultCode $expectCode]:"
    puts stderr $expectResult
    puts stderr "===="
}

# Test Procedure used by all tests
# id is the test identifier
# code is the test scenario
# optional -dontclean argument will stop the test classes being cleaned out
# alternatively -cleanup {script} will execute script before cleaning out

proc Test {id command expectCode expectResult args} {
    set resultCode [catch {uplevel $command} result]

    # debug
    puts "------------------------- Test $id:"

    if {($resultCode != $expectCode) ||
        ([string compare $result $expectResult] != 0)} {
        OutputTestError $id $command $expectCode $expectResult $resultCode \
                $result
    }

   if {[llength $args] == 0 || [lindex $args 0] != "-dontclean"} {
      if { [llength $args] > 0 && [lindex $args 0] == "-cleanup" } {
	  # post-completion cleanup specified
	  if { [llength $args] != 2 }  {
	      puts stderr "Test: -cleanup specified without cleanup script"
          } else {
	      set cscrpt [lindex $args 1]
 	      if {[catch {uplevel $cscrpt} rslt]} {
	          puts stderr "**Cleanup string \"$cscrpt\" returned \"$rslt\""
	      }
	  }
      }
   }
}

Test 1.1 {
    # max & min
    set rslt [list [max 1] [min 1]]
} 0 {1 1}

Test 1.2 {
    set rslt [list [max 1 2 3 2 1] [min 1 2 3 2 1]]
} 0 {3 1}

Test 1.3 {
    set rslt [list [max -4 -3 -2 -1] [min -4 -3 -2 -1]]
} 0 {-1 -4}

Test 1.4 {
    set rslt [list [max 4 3 2 1] [min 4 3 2 1]]
} 0 {4 1}

Test 1.5 {
    set rslt [list [max -1 -2 -3 -4] [min -1 -2 -3 -4]]
} 0 {-1 -4}

Test 1.6 {
    # Currently, max concats & evals its args
    set rslt [list [max "1 2 3" "45 56" 20] [min "1 2 3" "45 56" 20]]
} 0 {56 1}

Test 2.1 {
    # sum
    sum 3
} 0 {3}

Test 2.2 {
    sum 1 2 3 4
} 0 {10}

Test 2.3 {
    # Automatic switch to reals if any element is real
    sum 0.0 1 2 3 4
} 0 {10.0}

Test 2.4 {
    sum -0.1 3 -2.5
} 0 {0.4}

Test 2.5 {
    # Sum also concats args
    sum -0.4 "-0.1 3 -2.5" 5
} 0 {5.0}

Test 3.1 {
    # lwithout
    lwithout "1 2 3 4 5" "a"
} 0 {1 2 3 4 5}

Test 3.2 {
    lwithout "this is a list" "a"
} 0 {this is list}

Test 3.3 {
    lwithout "this is a longer list with two longer words in it" "longer"
} 0 {this is a list with two words in it}

Test 3.4 {
    lwithout "this is a longer list with two longer words in it" "*t*"
} 0 {this is a longer list with two longer words in it}

Test 3.5 {
    lwithout -glob "this is a longer list with two longer words in it" "*t*"
} 0 {is a longer longer words in}

Test 4.1 {
    # lwithoutl
    lwithoutl "1 2 3 4 5" "a"
} 0 {1 2 3 4 5}

Test 4.2 {
    lwithoutl "this is a list" "a"
} 0 {this is list}

Test 4.3 {
    lwithoutl "this is a longer list with two longer words in it" "longer"
} 0 {this is a list with two words in it}

Test 4.4 {
    lwithoutl "this is a longer list with two longer words in it" "*t*"
} 0 {this is a longer list with two longer words in it}

Test 4.5 {
    lwithoutl -glob "this is a longer list with two longer words in it" "*t*"
} 0 {is a longer longer words in}

Test 4.6 {
    lwithoutl "this is a longer list with two longer words in it" "list longer nonsense a"
} 0 {this is with two words in it}

Test 4.7 {
    lwithoutl "this is a longer list with two longer words in it" "l* *o is"
} 0 {this a longer list with two longer words in it}

Test 4.8 {
    lwithoutl -glob "this is a longer list with two longer words in it" "l* *o is"
} 0 {this a with words in it}

Test 4.9 {
    # reduce to nothing
    lwithoutl "this is a shorter list" "a longer list is this including shorter"
} 0 {}

Test 5.0 {
    # lassign - how should it work?
    lassign "1 2 3" one two three
    set rslt "$three $two $one"
} 0 {3 2 1}

Test 5.1 {
    # lassign with too few variables - don't complain
    lassign "a list of some words" out1 out2
    set rslt "$out2 $out1"
} 0 {list a}

Test 5.2 {
    # lassign with too many variables - set the extras to ""
    if {[info var out4] != ""} {unset out4}
    lassign "a list of" out1 out2 out3 out4
    set out4
} 0 {}

Test 5.3 {
    # lassign with embedded brace - leave it (brace is unescaped on parse)
    lassign "this contains a\{a brace" one two three four
    set three
    # Have to escape the brace that appears unescaped in the output...
} 0 "a\{a"

Test 5.4 {
    # lassign with illegal brace - error
    lassign "this contains \{a brace" one two three four
    set three
} 1 {unmatched open brace in list}

Test 5.5 {
    # lassign with proper sublist
    lassign "this contains \{a brace\}" one to three four
    set three
} 0 {a brace}

Test 6.1 {
    # lintersect
    lintersect "now is come the time for all good men" "to come to the party"
} 0 {come the}

Test 6.2 {
    # other way round
    lintersect "to come to the party" "now is come the time for all good men"
} 0 {come the}

Test 6.3 {
    # Repeated list elements don't get repeated in output
    lintersect "one two three four five four" "four five four"
} 0 {four five}

Test 6.4 {
    # Empty lists OK
    lintersect "one two three" "four five six"
} 0 {}

Test 6.5 {
    lintersect "one two three" ""
} 0 {}

Test 6.5 {
    lintersect "" "one"
} 0 {}

Test 7.1 {
    # nth - returns nth element of list argument, or nth argument
    nth 0 "one two three four"
} 0 {one}

Test 7.2 {
    nth 1 one two three four
} 0 {two}

Test 7.3 {
    nth -1 one two three four
} 0 {}

Test 7.4 {
    nth 3 one two three
} 0 {}

Test 7.5 {
    # This part is ugly.  It's just how it catenates its args
    nth 1 {one two three} four
} 0 {two}
Test 7.6 {
    nth 1 "{one two three}" four
} 0 {four}

Test 8.1 {
    # lmap
    lmap {{1 one} {2 two} {3} {4 four quatre}} "nth 1"
} 0 {one two {} four}

Test 8.2 {
    lmap "foo bar" "set dummy"
} 0 {foo bar}

Test 8.3 {
    lmap "" "set dummy"
} 0 {}

Test 9.1 {
    # cons
    cons 1 2
} 0 {1 2}

Test 9.2 {
    cons 1 "2 3 4"
} 0 {1 2 3 4}

Test 9.3 {
    cons {} 2
} 0 {{} 2}

Test 9.4 {
    cons "1 2" {}
} 0 {{1 2}}

Test 9.5 {
    cons 1 2 3 4
} 1 {called "cons" with too many arguments}

Test 10.1 {
    # reverse
    reverse "one two three four"
} 0 {four three two one}

Test 10.2 {
    reverse {{one two} {three four} five}
} 0 {five {three four} {one two}}

Test 10.3 {
    reverse one two three
} 1 {called "reverse" with too many arguments}

Test 10.4 {
    reverse {}
} 0 {}

Test 11.1 {
    # car
    car "one two three four"
} 0 {one}

Test 11.2 {
    car {{one two} three four}
} 0 {one two}

Test 11.3 {
    car {}
} 0 {}

Test 12.1 {
    # cdr
    cdr "one two three four"
} 0 {two three four}

Test 12.2 {
    cdr {{one two} three}
} 0 {three}

Test 12.3 {
    cdr {one}
} 0 {}

Test 12.4 {
    cdr ""
} 0 {}

Test 13.1 {
    # last
    last "one two three four"
} 0 {four}

Test 13.2 {
    last "one"
} 0 {one}

Test 13.3 {
    last ""
} 0 {}

Test 13.4 {
    last {one {}}
} 0 {}

Test 14.1 {
    # regmod
    regmod "foo" "befoobedoofoo" "bar"
} 0 {bebarbedoofoo}
    
Test 14.2 {
    regmod -all "foo" "befoobedoofoo" "bar"
} 0 {bebarbedoobar}
    
Test 14.3 {
    regmod -all ".oo" "befoobedoofoo" "bar"
} 0 {bebarbebarbar}

Test 15.1 {
    # stringreplace
    stringreplace "this is a string" 5 6 "was"
} 0 {this was a string}

Test 15.2 {
    stringreplace "this is a string" 7 6 " still"
} 0 {this is still a string}

Test 16.1 {
    # uptochr
    uptochr "t" "a long string with a t in it"
} 0 {a long s}

Test 16.2 {
    # uptochr
    uptochr ":" "a long string without a colon"
} 0 {a long string without a colon}

Test 17.1 { 
    # lunion - simple
    lunion "one two three" "four five"
} 0 {one two three four five}

Test 17.2 {
    # lunion - repeated elements
    lunion "now is the time for all" "good men to come to the aid"
} 0 {now is the time for all good men to come aid}

Test 17.3 {
    # lunion - other way around
    lunion "good men to come to the aid" "now is the time for all"
} 0 {good men to come the aid now is time for all}

Test 17.4 {
    # lunion - null parts
    lunion "one two" ""
} 0 {one two}

Test 17.5 {
    # lunion - other part null
    lunion "" "three four"
} 0 {three four}

} ;# End of test-utils block
