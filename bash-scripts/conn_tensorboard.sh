cd /mnt/personal/mrkosmic/synced/RL-cable/
ml tensorboard
fuser 16008/tcp
tensorboard --logdir=./experiments/logs --port=16008
TENSORBOARD_PID=$!

# Wait for TensorBoard to finish
wait $TENSORBOARD_PID
