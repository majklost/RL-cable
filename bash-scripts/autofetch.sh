#!/usr/bin/bash
#automatically fetch results from remote server

echo "Starting autofetch $$"

SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
#mock periodic fetch
while true
do
    #fetch results
    "$SCRIPTPATH/fetch_results.sh"
    echo "Fetching results"

    #wait for 120 seconds
    sleep $((60*5))
done
