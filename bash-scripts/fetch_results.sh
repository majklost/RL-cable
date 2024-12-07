#!/bin/bash
# Syncs the project to the remote server 
# Script called from the local machine
LOCAL_RESULTS="/home/michal/Documents/Skola/bakalarka/RL/RL-cable/experiments/"
REMOTE_RESULTS="/mnt/personal/mrkosmic/synced/RL-cable/experiments/"
REMOTE_USER="mrkosmic@login3.rci.cvut.cz"
echo "Syncing $LOCAL_RESULTS to $REMOTE_USER:$REMOTE_RESULTS"
rsync -avz    $REMOTE_USER:$REMOTE_RESULTS $LOCAL_RESULTS
echo "Done"
