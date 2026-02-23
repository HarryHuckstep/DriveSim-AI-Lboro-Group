#!/usr/bin/env python

import socket
import sys

def create_socket(timeout=1.0):
    """Creates and configures the UDP socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        return sock
    except socket.error as msg:
        print 'Could not make a socket:', msg
        sys.exit(-1)