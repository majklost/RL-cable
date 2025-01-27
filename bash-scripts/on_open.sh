#!/usr/bin/bash
#start tensorboard and start autopull of results
echo "Starting autofetch and tensorboard"
echo $$
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
# "$SCRIPTPATH/autofetch.sh" &
# "$SCRIPTPATH/autofetch.sh" &

# Start the SSH tunnel and TensorBoard in the background
cat "$SCRIPTPATH/conn_tensorboard.sh" | ssh mrkosmic@login3.rci.cvut.cz -L 16006:127.0.0.1:16008 &
SSH_PID=$!

cleanup() {
    echo "Cleaning up..."
    X=$(ssh mrkosmic@login3.rci.cvut.cz "fuser 16008/tcp")
    $(ssh mrkosmic@login3.rci.cvut.cz "kill $X")
}
trap cleanup EXIT
# Function to clean up background processes

# Trap SIGINT and SIGTERM to ensure cleanup is called

# Wait for background processes to finish
wait $SSH_PID
#kill children
# sleep 30
# pkill -P $$
