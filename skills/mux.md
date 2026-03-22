Start a long-running command in a background tmux session. If you are running multiple processes and have the opportunity to run many in parallel, try using mux to run them all at the same time, output to separate files, than gather the files output. Use the /tmp directory for outputting txt files.

After calling this tool, use run_bash to execute the tmux command:

  `tmux new-session -d -s {session_id} '{command}'`

Tips:
Check output:
  `tmux capture-pane -t {session_id} -p -S -`

Stop and clean up:
  `tmux capture-pane -t {session_id} -p -S - && tmux kill-session -t {session_id}`
