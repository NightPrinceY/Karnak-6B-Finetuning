#!/bin/bash
# One-shot check-and-relaunch, meant to be invoked periodically by Windows Task
# Scheduler via `wsl.exe -d Ubuntu -u elijah -- bash /path/to/this/script`.
# Unlike watchdog.sh (an infinite loop meant to run once, detached), this exits
# immediately after each check -- Task Scheduler owns the repeat interval.
# This runs from a fresh wsl.exe invocation each time, sharing no process
# ancestry with any Antigravity/IDE connection, so it survives even if
# whatever kills tmux/setsid-detached processes strikes again.

REPO=/home/elijah/src/Muslim-mode-finetuning
LOG="$REPO/watchdog.log"

log() { echo "$(date +'%Y-%m-%d %H:%M:%S') [task-scheduler-check] $1" >> "$LOG"; }

if ! pgrep -f "train/sft_lora.py" > /dev/null; then
    log "training process NOT found -- relaunching"

    tmux has-session -t muslim_train 2>/dev/null || tmux new-session -d -s muslim_train -c "$REPO"
    tmux has-session -t heartbeat 2>/dev/null || tmux new-session -d -s heartbeat

    if ! pgrep -f "sleep 30" > /dev/null; then
        tmux send-keys -t heartbeat "while true; do date +'%Y-%m-%d %H:%M:%S'; sleep 30; done >> $REPO/heartbeat.log" Enter
        log "heartbeat logger restarted"
    fi

    tmux send-keys -t muslim_train "cd $REPO && CUDA_VISIBLE_DEVICES=4 .venv/bin/python train/sft_lora.py 2>&1 | tee -a logs/run4_final.log" Enter
    log "relaunch command sent"
else
    log "training process OK, no action needed"
fi

# Also self-heal the in-WSL watchdog loop if IT died too
if ! pgrep -f "watchdog.sh$" > /dev/null; then
    log "in-WSL watchdog.sh also missing -- restarting it"
    setsid nohup "$REPO/watchdog.sh" < /dev/null > /dev/null 2>&1 &
    disown
fi
