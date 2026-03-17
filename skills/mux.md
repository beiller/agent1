Start a long-running command in a background tmux session.

After calling this tool, use run_bash to execute the tmux command:

  `tmux new-session -d -s {session_id} '{command}'`

Tips:
Check output:
  `tmux capture-pane -t {session_id} -p -S -`

Stop and clean up:
  `tmux capture-pane -t {session_id} -p -S - && tmux kill-session -t {session_id}`
