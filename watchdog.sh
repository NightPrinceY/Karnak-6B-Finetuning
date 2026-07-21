#!/bin/bash
# Auto-recovery watchdog: if the training process disappears (tmux server killed,
# process reaped, etc. -- without a full WSL2/VM reboot) this detects it within
# ~60s and relaunches training itself, resuming from the latest checkpoint,
# instead of waiting for a human to notice via "check progress".
# Launched detached from any tmux/session via setsid, so it does not share the
# same failure mode that has twice killed the tmux server while the VM stayed up.

REPO=/home/elijah/src/Muslim-mode-finetuning
LOG="$REPO/watchdog.log"

log() { echo "$(date +'%Y-%m-%d %H:%M:%S') $1" >> "$LOG"; }

log "watchdog started (pid $$)"

while true; do
    if ! pgrep -f "train/sft_lora.py" > /dev/null; then
        log "training process NOT found -- relaunching"

        # Ensure tmux server + sessions exist (may have been killed entirely)
        tmux has-session -t muslim_train 2>/dev/null || tmux new-session -d -s muslim_train -c "$REPO"
        tmux has-session -t heartbeat 2>/dev/null || tmux new-session -d -s heartbeat

        # (Re)start heartbeat logger if its loop isn't running
        if ! pgrep -f "sleep 30" > /dev/null; then
            tmux send-keys -t heartbeat "while true; do date +'%Y-%m-%d %H:%M:%S'; sleep 30; done >> $REPO/heartbeat.log" Enter
            log "heartbeat logger restarted"
        fi

        tmux send-keys -t muslim_train "cd $REPO && CUDA_VISIBLE_DEVICES=4 .venv/bin/python train/sft_lora.py 2>&1 | tee -a logs/run4_final.log" Enter
        log "relaunch command sent"

        # give it time to load before re-checking, avoid double-launch races
        sleep 180
    fi
    sleep 30
done
