#
# datefns.tcl
#
# Utility functions to parse and format date and time strings
# taken from my Alpha mail-reading package.
#
# dpwe@icsi.berkeley.edu 1996jan05
#

# Main functions are:
#	parseTime str - converts "10:30:11 pm" to "22 30 11"
#	parseDate str - converts 

proc now {} {
    # Return valid date/time string for now
    # Unix version
    return [exec date]
}

proc Warn {msg} {
    # Trap warnings under unix
    puts stderr "Warning: $msg"
}

proc val str {
# return the `value' of a string - 
# but since that is just a string anyway, really just 
# stripping leading zeros and nonnumerics
# i.e. "  009 " becomes "9"
	set rslt [string trim $str]
	set scale 0
	set c1 [string index $rslt 0]
	if { $c1 == "+" } {
		set scale 1
	} elseif { $c1 == "-" } {
		set scale -1
	}
	if { $scale == 0 } {
		set scale 1
	} else {
		# scale *was* set, so strip 1st chr off string
		set rslt [string range $rslt 1 end]
 	}
	set rslt [string trimleft $rslt "0"]
	if { [string length $rslt] == 0 }  {
		# stripped it *all* away - add something back!
		return "0"
	} else {
		if { $scale == -1 } {
			set rslt "-$rslt"
		}
		return $rslt
	}
}

proc parseTime str {
# find a field of form "hh:mm[:ss][ {am|pm}]" and return canonical
# "h m s" list.  Also return the indices of the located string
# first try for date followed by am/pm
	set timetxt ""
	# first look for colon-containing numeric string followed by am/pm
	set re "\[0-9:\]+:\[0-9:\]+\[ \t\]*\[AaPp\]\[Mm\]"
	if { [regexp -indices $re $str ixs] }  {
		# found time with am/pm
		set timetxt [string range $str [lindex $ixs 0] [lindex $ixs 1]]
		if { [regsub "\[ \t\]*\[Pp\]\[Mm\]" $timetxt "" timetxt ] } {
			# was pm 
			set houroffs 12
		} else {
			# must have been am - remove it
			if { [regsub "\[ \t\]*\[Aa\]\[Mm\]" $timetxt "" timetxt] == 0 } {
				Warn "parseTime: lost am in $timetxt"
				return 0
			}
		set houroffs 0
		}
		# should now have time string without am/pm
	} else {
		# that didn't work - try it without am/pm
		set re "\[0-9:\]+:\[0-9:\]+"
		if { [regexp -indices $re $str ixs] }  {
			# found time alone
			set timetxt [string range $str [lindex $ixs 0] [lindex $ixs 1]]
		}
	}
	if { [string length $timetxt] == 0 } {
		# no time found
		return 0
	}
	# else parse timetxt into hrs/mins/secs
	set time [split $timetxt ":"]
	# add on pm offset, if any 
	#  (have to take 'val' of hours to stop it being invalid octal if "08")
	set hours [val [lindex $time 0]]
	if { [info exists houroffs ] }  { 
		# i.e. we had an am/pm detected above
		if { $hours == 12 } {
			# 12 is dealt with specially 
			set hours 0
		} 
		set hours [expr "$houroffs + $hours"]
	}
	set mins [lindex $time 1]
	if { [llength $time] < 3 }  {
		# no seconds - set to zero
		set secs 0
	} else {
	 	set secs [lindex $time 2]
	}
	set time [list $hours [val $mins] [val $secs]]
	# return both time and indexes within string
	return [list $time $ixs]
}

set mail_tzTable "GMT 0 UT 0 CET 100 BST 100 EST -500 EDT -400 \
CST -600 CDT -500 MST -700 MDT -600 PST -800 PDT -700 JST 900 \
MET 100 MEST 200 MESZ 200 {WSU DST} 400"

# set DEFAULT_TIME_ZONE "PDT"

proc DefaultTimeZone {} {
    # Return the name of a default time zone to use, depending on the date
    # .. use whatever's returned by [now]
    set nowdate [parseDate [now]]
    return [lindex $nowdate 7]
}

proc tzNameToNum {n} {
	# convert a timezone name into corresponding offset from the table
	global mail_tzTable
	set ix [lsearch $mail_tzTable [string toupper $n]]
	if {$ix == -1} {
		Warn "tzNameToNum: zone '$n' unknown"
		return 0
	} else {
		return [lindex $mail_tzTable [expr "$ix+1"]]
	}
}

proc tzNumToName {t {dflt "num"}} {
    # convert an offset in minutes into a name (first found) or
    # format the number nicely
    # If no name found, $dflt governs behavior: if "num" (default)
    # substitute numeric representation.  Else return ""
    global mail_tzTable
    set ix [lsearch $mail_tzTable $t]
    if {$ix == -1} {
	if {$dflt == "num"} {
	    if { $t < 0 } {
		set t [expr -$t]
		set n "-[rpadLen "000$t" 4]"
	    } else {
		set n "+[rpadLen "000$t" 4]"
	    } 
	} else {
	    set n ""
	}		
    } else {
	set n [lindex $mail_tzTable [expr "$ix - 1"]]
    }
    return $n
}

proc parseDate {str {notime 0}} {
    # Take a string that is, e.g., the Date: field from a piece of email
    # return the date it represents in canonical numeric form: "1965 12 31"
    # If time component is find, return six element "1965 12 31 11 59 0"
    # seventh element is time zone as 100*signed hours + minutes rel GMT
    # (thus Eastern Standard Time is 500)
    # Will skip time parsing if notime is specified and nonzero
    # before starting, pull out any time field - defined as numeric 
    #  containing colons
    if {$notime==0} {
	set trtn [parseTime $str]
	if { [llength $trtn] > 1 } {
	    # did actually get a return
	    set time [lindex $trtn 0]
	    # time is now a three-element numeric list
	    set tlims [lindex $trtn 1]
	    # tlims is character indexes of time component
	    # strip out the time field from str
	    set str1 [string range $str 0 [expr "[lindex $tlims 0]-1"]]
	    set str2 [string range $str [expr "[lindex $tlims 1]+1"] end]
	    set str ${str1}${str2}
	} else {
	    set time [list 0 0 0]
	}
    }

    # first find month name
    set months [concat jan feb mar apr may jun jul aug sep oct nov dec]
    if {[regexp -indices -nocase [join $months "|"] $str indxs] == 0} {
	# no month name, so can only be "5/5/93" or "1993-5-5"
	if {[regexp "\[^0-9\]*(\[0-9\]*)\[^0-9\]*(\[0-9\]*)\[^0-9\]*(\[0-9\]*)" \
		$str dummy num1 num2 num3] }  {
	    if {[string length $num1] == 0 || [string length $num2] == 0 || \
		    [string length $num3] == 0} {
		# must find all three strings
		return ""
	    }
	    if {$num1 > 31}  {
		set year [val $num1]
		set monthIx [val $num2]
		set day [val $num3]
	    } else {
		if {$num1 > 12} {
		    # must be British format, dd/mm/yy, although most unreliable
		    set year [val $num3]
		    set monthIx [val $num2]
		    set day [val $num1]
		} else {
		    # assume american format: mm/dd/yy
		    set year [val $num3]
		    set monthIx [val $num1]
		    set day [val $num2]
		}
	    }
	} elseif {[regexp "\[^0-9\]*(\[0-9\]*)\[^0-9\]*(\[0-9\]*)" \
		$str dummy num1 num2] } {
	    # Maybe accept two numeric fields without year as default 
	    # to current year
	    set year 0
	    set monthIx [val $num1]
	    set day [val $num2]
	} else {
	    # no month and couldn't get three numeric fields
	    return ""
	}
    } else {
	# found month name
	set monthName [string tolower [string range $str [lindex $indxs 0] \
		[lindex $indxs 1] ]]
	set monthIx [expr "[lsearch $months $monthName] +1"]
	# now find tokens before and after month name
	set part [string range $str 0 [expr "[lindex $indxs 0] -1"]]
	set isbfr [regexp "(\[0-9\]*)\[^0-9]*$" $part dummy before]
	set part [string range $str [expr "[lindex $indxs 1] +1"] end]
	set isa1  [regexp "\[^0-9\]*(\[0-9\]*)" $part dummy after1]
	set isa2  [regexp "\[^0-9\]*\[0-9\]*\[^0-9\]*(\[0-9\]*)" $part dummy after2]
	if {$isa1 == 0 || [string length $after1] == 0} {
	    # no valid numeric after month name - not legal format (assume current year??)
	    if {!$isbfr} {
		# nothing before or after
		return ""
	    }
	    # For now, accept 14jun as being in current year
	    set year 0
	    set day [val $before]
	} elseif {$isbfr && [string length $before] > 0} {
	    # valid numeric prior to month: distinguish "1993mar03" and "03 mar 93"
	    if {$before > 31}  {
		set year [val $before]
		set day [val $after1]
	    } else {
		set year [val $after1]
		set day [val $before]
	    }
	} else {
	    # no 'before', so assume format "May 1st, 1907"
	    if {$isa2 == 0 || [string length $after2] == 0} {
		# must have two numeric fields after month if none before, else fail
		# return ""
		# For now, accept jun14 as shorthand for current year
		set year 0
		set day  [val $after1]
	    } else {
		set year [val $after2]
		set day  [val $after1]
		# As a hack, strip out the second post-month numeric field if 
		# we used it, so that dates with the year *after* the Timezone 
		# are still parsed correctly
		regsub $after2 $str "" str
		set str [string trim $str]
	    }
	}
    }
    # year, monthIx and day set up
	
    if {$year == 0} {
	# Year is set to zero if it is omitted and meant to default to current
	set year [lindex [parseDate [now]] 0]
    }
    if { $year < 100}  { set year [expr "$year + 1900"]  }
    if { [info exists time] } {
	# Check for timezone indication at end of string
	# three formats: "[+|-]0000" or "EST" or "-0700 (PST)"
	# also accept "UT" (universal time) from msn ??
	# must be at end
	if { [regexp {[+-][0-9][0-9][0-9][0-9]$} $str m] == 1 } {
	    set tz [val $m]
	    set tzname ""
	} elseif { [regexp {[ \t][A-z][A-z][A-z]?[A-z]?$} $str m] == 1 } {
	    # (matches 2 or 3 alpha characters preceded by whitespace)
	    # strip the delimiter off the front
	    set tzname [string range $m 1 end]
	    set tz [tzNameToNum $tzname]
	} elseif { [regexp {[+-][0-9][0-9][0-9][0-9] \([A-z ]+\)} \
		$str m] } {
	    # both - extract both
	    set tznum  [string range $m 0 4]
	    set tzname [string range $m 7 [expr [string len $m]-2]]
	    #	Warn "tznum:'$tznum'; tzname:'$tzname'"
	    # use the num field
	    set tz [val $tznum]
	} elseif { [regexp {([A-Z][A-Z][A-Z])([+-][0-9]+)([A-Z]*)} \
		$str m tzname tzoffs tzofnam] } {
	    # was something like "MET+1MEST" which I have seen - 
	    # treat as MET
	    set tz [tzNameToNum $tzname]
	} elseif { [regexp {\(([A-Z ]+)\)$} \
		$str m tzname] } {
	    # accept (MET DST) as "MET DST"
	    set tz [tzNameToNum $tzname]
	} else {
	    # No timezone spec found
	    set tzname [DefaultTimeZone]
	    set tz [tzNameToNum $tzname]
	    if {$time != "0 0 0"} {
		Warn "couldn't find a tzname in '$str'"
	    }
	}
	return [concat $year $monthIx $day $time $tz $tzname]
    } else {
	# no time found
	return [concat $year $monthIx $day]
    }
}

proc shortDate date {
# convert canonical numeric date "1993 12 31" into short "dec31" format
	set months [concat jan feb mar apr may jun jul aug sep oct nov dec]
	set rslt [lindex $months [expr "[lindex $date 1] - 1"]]
	set day  [lindex $date 2]
	if {$day < 10} {
		set rslt "${rslt}0${day}"
	} else {
		set rslt "${rslt}${day}"
	}
	return $rslt
}

proc danDate date {
# convert canonical date "1993 12 31" to 'dan' format "1993dec31"
	return "[lindex $date 0][shortDate $date]"
}

proc timeFmtSecs {secs} {
    # format a number of seconds as an [hh:]mm:ss string
    set sign ""
    if {$secs < 0} {
	set secs [expr -$secs]
	set sign "-"
    }
    set output [format ":%02d" [expr $secs%60]]
    set mins [expr $secs/60]
    set hrs  [expr $mins/60]
    if {$hrs != 0} {
	set output [format "%d:%02d" $hrs [expr $mins%60]]$output
    } else {
	if {$mins != 0} {
	    set output $mins$output
	}
    }
    return $sign$output
}

proc timeFmt time {
# convert two or three element list into "20:29" or "20:29:13"
	set l [llength $time]
	if { $l != 2 && $l != 3 }  {
		Warn "timeFmt: $time is not two or three element list"
		return
	}
	set rslt ""
	for { set i 0 } { $i < $l } { incr i } {
		set x [lindex $time $i]
		if { $x < 10 }  {
			set rslt "${rslt}0$x"
		} else {
			set rslt "${rslt}$x"
		}
		if { $i < [expr "$l - 1"]  }  {
			set rslt "$rslt:"
		}
	}
	return $rslt
}

proc CanonicalDays date {
# calculate canonical days since jan1 1970
	set monthDays "31 28 31 30 31 30 31 31 30 31 30 31"
	set monthDaysCum "0"
	set x 0
	for {set i 0} {$i<12} {incr i} {
		incr x [lindex $monthDays $i]
	    set monthDaysCum "$monthDaysCum $x"
	}
	set year  [lindex $date 0]
	set month [lindex $date 1]
	set day   [lindex $date 2]
	set ndays  [expr "( ( 1461 * ( $year - 1970 ) + 1) / 4 )"]
# add 1 before dividing by 4 because 1972 is the first leap year - 
# want to make sure the .25 * year adds one for 1973
	set isLeapYear [expr "($year % 4) == 0"]
	set ndays [expr "$ndays + [lindex $monthDaysCum [expr "$month - 1"]] \
						+ ($isLeapYear && ($month > 2))"]
	set ndays [expr "$ndays + $day - 1"]
	return $ndays
}

proc InvCanonicalDays {days} {
    # Convert canonical days back into {$year $month $day} list
    # start from 1969 instead, so that nfouryears==1 for 1973jan01
    set nfouryears [expr ($days+365)/(4*365+1)]
    set within4yr  [expr ($days+365)%(4*365+1)]
    set yrwithin4  [expr $within4yr/365]
    if {$yrwithin4 > 3} {
	set yrwithin4 3
	set yearday   365
    } else { 
	set yearday [expr $within4yr%365]
    }
    set year [expr 1969+4*$nfouryears+$yrwithin4]
    set isleap  [expr $yrwithin4==3]

# puts stderr "year=$year yearday=$yearday"
    set monthDays "31 28 31 30 31 30 31 31 30 31 30 31"
    set x 0
    set month 0
    for {set i 0} {$i<12} {incr i} {
	set y $x
	incr x [lindex $monthDays $i]
	# add extra feb day
	if {$isleap && $i==1} {incr x}
	if {$yearday >= $y && $yearday < $x} {
	    set month [expr 1+$i]
	    set day [expr 1+$yearday-$y]
	    break
	}
    }
    return [list $year $month $day]
}

proc DayOfWeek date {
# return 0..6 corresponding to Sun..Sat for the given numeric date
	set ndays [CanonicalDays $date]
	return [expr "($ndays + 4) % 7"]
# jan 1st 1970 was a thursday (of course) - so add 4
# works for christmas 1992 & today (1994 2 9)
}

proc Capitalize str {
# make first chr of str a capital letter
	set first [string index $str 0]
	set rest  [string range $str 1 end ]
	return "[string toupper $first]$rest"
}

proc medTime {time {showTZ 1}} {
# take canonical time format "18 52 25 500 PDT" and convert to "18:52:25 +0500 (EST)"
# Time zone is omitted if !$showTZ
    set len [llength $time]
    set hrs [lindex $time 0]
    set min [lindex $time 1]
    set sec 0
    set tz 0
    set tzname ""
    if { $len > 2 }  {
	# more than just hrs and min
	set sec [lindex $time 2]
	if { $len > 3 }  {
	    set tz [lindex $time 3]
	    if {$len > 4} {
		set tzname [lindex $time 4]
	    }
	}
    }
    set rslt "[rpadLen "0$hrs" 2]:[rpadLen "0$min" 2]:[rpadLen "0$sec" 2]"
    if {$tzname == ""} {
	set tzname [tzNumToName $tz none]
    }
    if {$tzname != ""} {
	set tzname " ($tzname)"
    }
    set tzs [expr ($tz<0)?"-":"+"]
    set atz [expr abs($tz)]
    if {$showTZ} {
	return [concat $rslt $tzs[format %04d $atz]$tzname]
    } else { 
	return $rslt
    }
}

proc medDate {date {showTZ 1}} {
# convert canonical numeric date "1993 12 31" into medium format e.g.
# "Mon, 31 Jan 94 18:52:35 +0400 (EST)"  Timezone spec omitted if !$showTZ
	set weekdays [concat sun mon tue wed thu fri sat]
	set months [concat jan feb mar apr may jun jul aug sep oct nov dec]
	set rslt [Capitalize [lindex $weekdays [DayOfWeek $date]]]
	set year  [lindex $date 0]
	set month [lindex $date 1]
	set day   [lindex $date 2]
	set rslt "${rslt}, $day [Capitalize [lindex $months [expr ${month}-1]]]"
	set rslt "${rslt} [rpadLen "00[expr ${year}%100]" 2]"
	if { [llength $date] > 3 } {
	  # date has more than three fields -> must have time too
	  set rslt "${rslt} [medTime [lrange $date 3 end] $showTZ]"
	}
#	return "${rslt}."
	return "${rslt}"
}
	
proc padLen {str len} {
# truncate str, or append spaces, to make exactly <len> long
	set spaces "                                                        "
	set slen [string length $str]
	set rslt $str
	if {$slen > $len}  {
		set rslt [string range $rslt 0 [expr "$len - 1"]]
	} else {
		if {$slen < $len} {
			set rslt ${rslt}[string range $spaces 0 [expr "$len - $slen - 1"]]
		}
	}
	return $rslt
}

proc rpadLen {str len} {
# truncate str FROM END, or PREpend spaces, to make exactly <len> long
	set spaces "                "
	set slen [string length $str]
	set rslt $str
	if {$slen > $len}  {
		set rslt [string range $rslt [expr "$slen - $len"] end]
	} else {
		if {$slen < $len} {
			set rslt [string range $spaces 0 [expr "$len - $slen - 1"]]${rslt}
		}
	}
	return $rslt
}

proc Date2Secs {date} {
    # Convert a date list into seconds since the epoch
    # date is {year month day hour min sec [timezone]}
    set days [CanonicalDays $date]
    set hrs [lindex $date 3]
    set min [lindex $date 4]
    set sec [lindex $date 5]
    if {[llength $date]>6} {
	# has timezone - add it to hour
	set tznum [lindex $date 6]
	if {$tznum > 0} {
	    incr hrs [expr -$tznum/100]
	    incr min [expr -$tznum%100]
	} else {
	    set tznum [expr -$tznum]
	    incr hrs [expr $tznum/100]
	    incr min [expr $tznum%100]
	}
    }
    set secs [expr $sec+60*($min+60*($hrs+24*$days))]
}

proc Secs2Date {secs {tz ""}} {
    # Convert a seconds count into a formatted date string
    # Add in the timezone effect
    if {$tz == ""} {
	set tz [DefaultTimeZone]
    }
    set tznum [tzNameToNum $tz]
    if {$tznum > 0} {
	incr secs [expr 60*($tznum/100)*60+($tznum%100)]
    } else {
	set tzn [expr -$tznum]
	incr secs [expr -60*(($tzn/100)*60+($tzn%100))]
    }
    set secsperday [expr 60*60*24]
    set days [expr $secs/$secsperday]
    set ofsecs [expr $secs%$secsperday]
    set ymd [InvCanonicalDays $days]
    set secs [expr $ofsecs%60]
    set mins [expr ($ofsecs/60)%60]
    set hrs  [expr ($ofsecs/3600)]
    lappend ymd $hrs $mins $secs $tznum $tz
    return $ymd
}