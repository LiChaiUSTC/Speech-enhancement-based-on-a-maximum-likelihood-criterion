#
# rdprocmgr.tcl
#
# Routines to provide central management of readprocs
# .. needed for integ_frontend
#
# 1997sep25 dpwe@icsi.berkeley.edu
# $Header: /n/yam/da/dpwe/projects/sprachdemo/RCS/rdprocmgr.tcl,v 1.1 1999/01/12 08:26:13 dpwe Exp $

if {[info var _rdprocs] == ""} {
    # Only define this stuff once

    set _rdprocs {}

    set _rdprocsEnabled 0

    proc rdproc_TurnOn {pipe proc} {
#	puts stderr "**rdp_On: $pipe proc='$proc'"
	fileevent $pipe readable $proc
    }

    proc rdproc_TurnOff {pipe} {
#	puts stderr "**rdp_Off: $pipe"
	fileevent $pipe readable {}
    }

    proc rdprocAdd {pipe proc} {
	# Register that, when enabled, $proc should be invoked for pipe $pipe
	global _rdprocs _rdprocsEnabled
#	puts stderr "rdprocAdd: $pipe proc='$proc' (rdprocs='$_rdprocs')"
	set newprocs {}
	set found 0
	foreach pair $_rdprocs {
	    lassign $pair opipe oproc
	    if {$opipe == $pipe} {
		Warn "rdprocAdd: replacing '$oproc' with '$proc' for $opipe"
		set oproc $proc
		if {$_rdprocsEnabled} {
		    # disable the old proc if it's active
		    rdproc_TurnOff $opipe
		}
		set found 1
	    }
	    lappend newprocs [list $opipe $oproc]
	}
	if {!$found} {
	    lappend newprocs [list $pipe $proc]
	}
	# Enable the new proc if we're online
	if {$_rdprocsEnabled} {
	    rdproc_TurnOn $pipe $proc
	}
	set _rdprocs $newprocs
    }

    proc rdprocRemove {pipe} {
	# Remove the reader and the named pipe
	global _rdprocs _rdprocsEnabled
#	puts stderr "rdprocRemove: $pipe"
	set newprocs {}
	set found 0
	foreach pair $_rdprocs {
	    lassign $pair opipe oproc
	    if {$opipe == $pipe} {
		set found 1
		if {$_rdprocsEnabled} {
		    # disable the proc if it's active
		    rdproc_TurnOff $opipe
		}
	    } else {
		lappend newprocs [list $opipe $oproc]
	    }
	}
	if {!$found} {
	    Warn "rdprocRemove: pipe $pipe not found"
	}
	set _rdprocs $newprocs
    }

    proc rdprocEnable {} {
	# Attach all the registered _rdprocs
	global _rdprocs _rdprocsEnabled
#	puts stderr "rdprocEnable (rdprocs='$_rdprocs')"
	if {$_rdprocsEnabled == 0} {
	    foreach pair $_rdprocs {
		lassign $pair opipe oproc
		rdproc_TurnOn $opipe $oproc
	    }
	}
	set _rdprocsEnabled 1
    }

    proc rdprocDisable {} {
	# Detach _rdprocs for all registered pipes
	global _rdprocs _rdprocsEnabled
#	puts stderr "rdprocDisable"
	if {$_rdprocsEnabled != 0} {
	    foreach pair $_rdprocs {
		lassign $pair opipe oproc
		rdproc_TurnOff $opipe
	    }
	}
	set _rdprocsEnabled 0
    }

    proc rdprocClosePipe {pipe} {
	# Close the pipe & remove any rdproc
#	puts stderr "rdprocClosePipe: $pipe"
	rdprocRemove $pipe
	close $pipe
    }

    # Close the conditional execution
}
