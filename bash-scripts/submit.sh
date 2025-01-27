#!/usr/bin/bash

# Syncs the project to the remote server
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
"$SCRIPTPATH/to_remote.sh" # Syncs the project to the remote server

# Submit simple 1 core job
cat "$SCRIPTPATH/remote_submit.sh" | ssh mrkosmic@login3.rci.cvut.cz 

