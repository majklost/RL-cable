#!/bin/bash
# Syncs the project to the remote server 
# Script called from the local machine
LOCAL_PROJECT_DIR="/home/michal/Documents/Skola/bakalarka/RL/RL-cable/"
REMOTE_PROJECT_DIR="/mnt/personal/mrkosmic/synced/RL-cable/"
REMOTE_USER="mrkosmic@login3.rci.cvut.cz"
echo "Syncing $LOCAL_PROJECT_DIR to $REMOTE_USER:$REMOTE_PROJECT_DIR"

rsync --delete -avz --update --exclude='experiments/*' --exclude='__pycache__' --exclude='.git' --exclude='.idea' --exclude='.venv' --exclude='.venv_old' --exclude='.vscode' $LOCAL_PROJECT_DIR $REMOTE_USER:$REMOTE_PROJECT_DIR
rsync -avz --update \
  $LOCAL_PROJECT_DIR/experiments/ \
  $REMOTE_USER:$REMOTE_PROJECT_DIR/experiments/
echo "Done"
