#
# webfns.tcl
#
# Tcl functions for web pages - from filter-html.cgi
#
# dpwe@media.mit.edu 1997may14
# $Header: /n/yam/da/dpwe/lib/share/tcl/RCS/webfns.tcl,v 1.2 1997/08/07 16:39:03 dpwe Exp $


# Uses getenv from utilfns.tcl, for echoing HTTP fields in request
proc GetURL {url} {
    # Return the text of a URL
    global sitepath pagepath
    if {[regexp {([^:]*)://([^:/]*)(:([0-9]*))?(.*)} $url \
	    all type host dummy port page] == 0} {
	# Maybe skipped the http: part...
	if {[regexp {([^:/]*)(:([0-9]*))?(.*)} $url \
		all host dummy port page] == 0} {
	    return "<HTML><HEAD><TITLE>GetURL Error</TITLE></HEAD><BODY><H1>GetURL Error:</H1><P>Couldn't parse \"$url\" as a URL</P></BODY></HTML>"
	}
	# Else, was a URL without a transport type
	set type "http"
    }
    if {$port == ""} {
	# default port
	set port 80
    }
    if {$page == ""} {
	# default page
	set page "/"
    }

    # Open a socket
    set sk [socket $host $port]
    # Send the request
    puts $sk "GET $page HTTP/1.0"
    # Echo through any client-provided fields
    if {[set e [getenv "HTTP_USER_AGENT"]] != ""} {puts $sk "User-Agent: $e"}
    if {[set e [getenv "HTTP_HOST"]] != ""} {puts $sk "Host: $e"}
    if {[set e [getenv "HTTP_ACCEPT"]] != ""} {puts $sk "Accept: $e"}
    # Indicate end-of-header, make sure it's all delivered
    puts $sk ""
    flush $sk
    # Get the reply
    set text [read $sk]
    close $sk

    # Parse the returned header
    set hbrk [string first "\n\n" $text]
    if {$hbrk < 0} {
	Error "Couldn't parse reply from $url:\n$text"
    }
    set hedr [string range $text 0 [expr $hbrk-1]]
    set body [string range $text [expr $hbrk+1] e]

    # Handle "Location:" header silently
    if {[regexp "\nLocation: *(\[^\n\]*)" $hedr all loc]} {
	# There was a redirect
	Warn "redirect to $loc"
	return [GetURL $loc]
    }

    # Set up the global variables for rebuilding relative references
    # from this page
    set sitepath "$type://$host$dummy"
    if {[string range $page e e] == "/"} {
	# page ends in slash - is its own directory
	set pagepath "$sitepath$page"
    } else {
	# strip back to containing directory
	set pagepath "$sitepath[file dir $page]/"
    }

    return $body
}
