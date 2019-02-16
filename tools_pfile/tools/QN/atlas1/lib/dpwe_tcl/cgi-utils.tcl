#
# form-utils.tcl
#
# Define some Tcl utility functions for the processing of HTML forms 
# from within Tcl CGI scripts.
#
# dpwe@icsi.berkeley.edu 1996jun05
# $Header: /u/drspeech/src/guitools/dpwetcl/RCS/cgi-utils.tcl,v 1.6 1998/08/12 18:28:09 dpwe Exp $

# Package header
#set dpwe_cgiutils_vsn 0.1
#package provide "Dpwe_CgiUtils" $dpwe_cgiutils_vsn
# I think the package provide bit is done by the pkgIndex code.

# Since we're being sourced by all these cgi scripts, may as well
# put the hard-coded path in here.  Heck, I could even configure 
# it if I wanted to.

# First, get the machine independent lib.  This lets us figure out the 
# machine type
#set dpwe_prefix "/u/dpwe/share"
#lappend auto_path "$dpwe_prefix/lib"
if {[catch {set SPEECH_DIR $env(SPEECH_DIR)}]} {
    if {[catch {set SPEECH_DIR [glob "~drspeech"]}]} {
	# Default so error is file not found rather than uninitialized var
	setenv SPEECH_DIR "/u/drspeech"
	foreach dir {"/homes/drspeech" "/u/drspeech"} {
	    if {[file exists $dir]} {
		set SPEECH_DIR $dir
		break
	    }
	}
    }
}

set sharelibdir "$SPEECH_DIR/share/lib"
lappend auto_path $sharelibdir
#puts stderr "auto_path=$auto_path"
catch {package require Dpwe_Utilfns}
#set machtype [exec /u/drspeech/share/bin/speech_arch]
set machtype [SpeechArch]

# Now point to my arch-dep libraries.
# .. then here it is
#set dpwe_exec_prefix "/u/dpwe/$machtype"
#lappend auto_path "$dpwe_exec_prefix/lib"
regsub "share" $sharelibdir $machtype machlibdir
lappend auto_path $machlibdir
# load binary file-reading extension (breadi, bfilecopy etc).
package require Dpweutils_tcl

proc plus2spc {str} {
    # replace "+" with " " in string
    set rslt $str
    regsub -all {\+} $str " " rslt
    return $rslt
}

proc hex2chr {str {escape "%"} {fixBraces 1}} {
    # replace occurrences of %nn with single characters
    # Splat unconverted escapes while scanning, but track & restore later
    # Unless fixBraces is zero, map {, } -> \{, \}
    set unconverted {}
    while {[set ix [string first $escape $str]] > -1} {
	set hex [string range $str [expr $ix+1] [expr $ix+2]]
	if {[scan $hex "%x" num]} {
	    # successfully converted to chr - replace
	    set ch [format "%c" $num]
	    # escape braces and backslashes (for tcl handling)
	    if {$ch == "\{" || $ch == "\}" || $ch == "\\"} {
		set ch "\\$ch"
	    }
	    set str [stringreplace $str $ix $ix+2 $ch]
	} else {
	    # couldn't convert - but hide it so we don't find it again
	    set str [stringreplace $str $ix $ix "*"]
	    lappend unconverted $ix
	}
    }
    # remap all the unconverted escapes
    foreach ix $unconverted {
	set str [stringreplace $str $ix $ix $escape]
    }
    return $str
}

proc CheckPostEnvironment {} {
    # Performs some checks to confirm that this script was accessed with 
    # a POST command.  Returns "" if OK, else an error message.
    if {[getenv REQUEST_METHOD] != "POST"} {
	return "This script should be referenced with a METHOD of POST (not [getenv REQUEST_METHOD])."
    }
    set ctype [getenv CONTENT_TYPE]
    if {$ctype == "application/x-www-form-urlencoded"} {
	return "urlencoded"
    } elseif {[string match "multipart/form-data;*" $ctype]} {
	set boundary ""
	regexp {boundary=([^;]*);?} $ctype all boundary
	return [list "multipart" $boundary]
    } else {
	return [list "error" "This script can only be used to decode form results (not CONTENT_TYPE $ctype)."]
    }
}

proc CheckGetEnvironment {} {
    # Performs some checks to confirm that this script was accessed with 
    # a GET command.  Returns "" if OK, else an error message.
    if {[getenv REQUEST_METHOD] != "GET"} {
	return "This script should be referenced with a METHOD of GET (not [getenv REQUEST_METHOD])."
    }
    if {[set ctype [getenv CONTENT_TYPE]] != ""} {
	return "This script can only be used to decode form results (not CONTENT_TYPE $ctype)."
    }
    return ""
}

proc oldUplevelVars {varnames {level 1}} {
    # Copy a list of variables to the same values in a given context
    global tcl_patchLevel
    puts stderr "varnames=$varnames level=$level tcl=$tcl_patchLevel"
    foreach v $varnames {
	# escape all braces not preceded by a brace?
#	set cmd "set uvTmp \[set $v\]; regsub -all {(\[^\\\])(\[\{\}\])} \$uvTmp \{\\1\\\\\\2\} uvTmp; uplevel $level \"set $v \\\{\$uvTmp\\\}\""
	set cmd "set uvTmp \[set $v\]; puts stderr \[set uvTmp\]; regsub -all {(\[^\\\])(\[\{\}\])} \$uvTmp \{\\1\\\\\\2\} uvTmp; uplevel $level \"set $v \$uvTmp\""
	puts stderr $cmd
	uplevel 1 $cmd
	# uplevel 1 "uplevel $level \"set $v \\\{\[set $v\]\\\}\""
    }
}

proc oldSplitAndAssign {line {separator "&"}} {
    # Process a line of the form "a=b&c=d" by breaking into pieces 
    # on $separator, then setting tcl vars whose names are given by the 
    # LHS of the "=" to the values on the RHS.  Apply usual HTTP 
    # translations.  Variables are set in caller's context
    set message [split $line $separator]
    set varnames {}
    foreach m $message {
	# remove excess spaces
	set m [string trim $m]
	set varconts [hex2chr [split [plus2spc $m] "="]]
	set varname [lindex $varconts 0]
	set varval [string trim [lindex $varconts 1]]
	# If the value is completely enclosed in quotes, strip them
	regexp {^"(.*)"$} $varval dummy varval
	set $varname $varval
	lappend varnames $varname
    }
    UplevelVars $varnames
    return $varnames
}

proc oldGetPostVars {} {
    # Handle the 'content' of the FORMS post and copy the values into 
    # equivalently-named Tcl variables.  Returns a list of the Tcl 
    # variable names that have been set.
    set line [read stdin [getenv CONTENT_LENGTH]]
    set varnames [SplitAndAssign $line "&"]
    # Copy the variable values into the calling context
    UplevelVars $varnames
    return $varnames
}

proc GetSimpleGetVars {} {
    # Handle the content of a GET post i.e. a string with '?'s separating
    set message [hex2chr [plus2spc [split [getenv QUERY_STRING] "?"]]]
    return $message
}

proc oldGetGetVars {} {
    # Assign vars set via FORM..GET i.e. in QUERY_STRING
    set varnames [SplitAndAssign [getenv QUERY_STRING] "&"]
    UplevelVars $varnames
    return $varnames
}


proc SplitAndAssignArr {arrayname line {separator "&"}} {
    # Process a line of the form "a=b&c=d" by breaking into pieces 
    # on $separator, then setting tcl vars whose names are given by the 
    # LHS of the "=" to the values on the RHS.  Apply usual HTTP 
    # translations.  Variables are set in caller's context
    upvar 1 $arrayname vararray
    set message [split $line $separator]
    set varnames {}
    foreach m $message {
	# remove excess spaces
	set m [string trim $m]
	set varconts [hex2chr [split [plus2spc $m] "="]]
	set varname [lindex $varconts 0]
	set varval [string trim [lindex $varconts 1]]
	# If the value is completely enclosed in quotes, strip them
	regexp {^"(.*)"$} $varval dummy varval
	lappend varnames $varname
	# Store in the uplevel array
	set vararray($varname) $varval
	#set $varname $varval
    }
    return $varnames
}

proc GetPostVarsArr {arrayname} {
    # Handle the 'content' of the FORMS post and copy the values into 
    # equivalently-named Tcl variables.  Returns a list of the Tcl 
    # variable names that have been set.
    upvar 1 $arrayname vararray
    set line [read stdin [getenv CONTENT_LENGTH]]
    set varnames [SplitAndAssignArr vararray $line "&"]
    return $varnames
}

proc GetGetVarsArr {arrayname} {
    # Assign vars set via FORM..GET i.e. in QUERY_STRING
    upvar 1 $arrayname vararray
    set varnames [SplitAndAssignArr vararray [getenv QUERY_STRING] "&"]
    return $varnames
}

proc GetMultipartSimpleVar {inpipe boundary} {
    # Read a simple named var's value from a multipart post
    set done 0
    set rslt ""
    while {!$done && ![eof $inpipe]} {
	set l [gets $inpipe]
	# Again, appends "--" to final separator
	if {$l == $boundary || $l == "$boundary--"} {
	    set done 1
	} elseif {$l != ""} {
	    if {$rslt == ""} {
		set rslt $l
	    } else {
		append rslt "\n$l"
	    }
	}
    }
    if {!$done} {
	set rslt "ERROR: GetSimpleMult: '$rslt' didn't terminate"
    }
    return $rslt
}

proc GetMultipartFileVar {inpipe boundary filename} {
    # copy it binarily
    if {[set outfile [open $filename "w"]] == ""} {
	return "ERROR: couldn't create '$filename' for output"
    }
    fconfigure $inpipe -translation binary -buffering none
    fconfigure $outfile -translation binary -buffering full
    set nbytes [bfilecopy $inpipe $outfile -1 "\r\n$boundary"]
    close $outfile
    fconfigure $inpipe -translation auto -buffering line
    # variable will be set to detail record
    return [list "FILE" $filename]
}

proc GetMultipartPart {inpipe boundary savedir} {
    # Retrieve one field of a multipart message from $inpipe
    # Read up to the next occurrence of $boundary
    # save any type=FILE data to $savedir
    # return the names of the global variables written to
    # or "" if it's all done
    # Skip blank lines. return on EOF cruft
    while {![eof $inpipe] && [set l [gets $inpipe]] == ""} 	{}
    if {[eof $inpipe] || $l == "--"} 				{return ""}
    # First header line must be Content-Disposition
    if {![regexp "Content-Disposition: *(\[^;\]*) *;? *\(.*\) *$" \
	    $l all disposition rest]} {
	return "ERROR: Block starts with '$l', not Content-Disposition"
    }
    set fldnames [SplitAndAssign fnarr $rest ";"]
    if {$disposition != "form-data"} {
	return "ERROR: Content-Disposition '$disposition', not form-data"
    }
    if {[lsearch $fldnames "name"] < 0} {
	return "ERROR: no 'name' field in '$l'"
    }
    set name $fnarr(name)
    set type ""
    # Read rest of header lines through to blank separator line
    while {[set l [gets $inpipe]] != ""} {
	if {[regexp "Content-Type: *(\[^; \]*)" $l all type]} {
	    # recognized content-type
	} else {
	    return "ERROR: unknown header line: '$l'"
	}
    }
    # OK, now we have seen the post-header blank line - read the 
    # data through to next $boundary & write result into var named $name
    if {[info vars $name] == ""} {
	global $name
	set $name ""
    }
    # default type is simple text
    if {$type == "" } {
	set val [GetMultipartSimpleVar $inpipe $boundary]
    } elseif {[lsearch $fldnames "filename"]} {
	set filename $fnarr(filename)
	set val [GetMultipartFileVar $inpipe $boundary "$savedir/$filename"]
    } else {
	return "ERROR: unknown type '$type' is unknown & no filename"
    }
    if {[string match "ERROR:*" $val]} {
	return $val
    } elseif {$type != ""} {
	lappend val $type
    }
    if {[set $name] == ""} {
	set $name $val
    } else {
	lappend $name $val
    }
    return $name
}

proc GetPostMultipart {inpipe boundary {savedir "."}} {
    # Form was CONTENT_TYPE multipart/form-data - check out the parts
    # TYPE=FILE fields are uploaded to their given name, below $savedir
    set varnames ""
    set l [gets $inpipe]
    # for some reason, there are an extra two dashes preprended to 
    # boundary ??
    set boundary "--$boundary"
    if {$l != $boundary} {
	return "ERROR: content first line '$l' doesn't match bdry '$boundary'"
    }
    set done 0
    while {!$done && ![eof $inpipe]} {
	set name [GetMultipartPart $inpipe $boundary $savedir]
	if {[string match "ERROR:*" $name]} {
	    return $name
	} elseif {$name == ""} {
	    set done 1
	} else {
	    lappend varnames $name
	}
    }
    return $varnames
}

proc ErrorExit {msg} {
    # Produce a fully-formatted output of an error page
    global argv0 tcl_version

    puts "Content-type: text/html\n"

    puts "<HTML>"
    puts "<HEAD>"
    set name $argv0
    regsub {.*/} $name "" name
    puts "<TITLE>$name OUTPUT</TITLE>"
    puts "</HEAD>"

    puts "<BODY BGCOLOR=\"#000000\" TEXT=\"#FFFFFF\" LINK=\"#0000FF\" VLINK=\"#0033FF\" ALINK=\"#0011FF\">"
    puts "<H1>$name Output:</H1>"
    puts "<P>(Tcl-version is $tcl_version)</P>"

    puts "<H1>Error:</H1>\n<pre>$msg</pre>"
    puts "</BODY>"
    puts "</HTML>"
    
    exit 1
}
