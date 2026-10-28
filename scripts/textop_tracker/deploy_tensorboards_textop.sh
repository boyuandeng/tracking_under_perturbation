#!/usr/bin/env bash
# Run locally. Every ssh invocation uses the host alias and ~/.ssh/config as-is.
set -Eeuo pipefail

REMOTE=${REMOTE:-deploy}
CONTAINER=${CONTAINER:-stablemimic-zzy-gpu23-textop}
ISAAC_PORT=${ISAAC_PORT:-6009}
MJLAB_PORT=${MJLAB_PORT:-6007}
LOCAL_ISAAC_PORT=${LOCAL_ISAAC_PORT:-$ISAAC_PORT}
LOCAL_MJLAB_PORT=${LOCAL_MJLAB_PORT:-$MJLAB_PORT}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-120}
OPEN_BROWSER=${OPEN_BROWSER:-1}

command -v ssh >/dev/null || { echo 'ssh is required.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'local python3 is required.' >&2; exit 1; }
if [[ -z "$REMOTE" || "$REMOTE" == -* || ! "$CONTAINER" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo 'Invalid REMOTE or CONTAINER.' >&2
    exit 1
fi
for value in "$ISAAC_PORT" "$MJLAB_PORT" "$LOCAL_ISAAC_PORT" "$LOCAL_MJLAB_PORT" "$STARTUP_TIMEOUT"; do
    if [[ ! "$value" =~ ^[1-9][0-9]{0,4}$ ]] || (( 10#$value > 65535 )); then
        echo 'Ports and STARTUP_TIMEOUT must be integers between 1 and 65535.' >&2
        exit 1
    fi
done
if [[ "$ISAAC_PORT" == "$MJLAB_PORT" ]]; then
    echo 'ISAAC_PORT and MJLAB_PORT must differ.' >&2
    exit 1
fi

STATE_DIR=$(mktemp -d /tmp/deploy-tensorboards.XXXXXXXX)
RUN_ID="deploy-tensorboards-$(date +%Y%m%d-%H%M%S)-${STATE_DIR##*.}"
SSH_PID=''
REMOTE_ATTEMPTED=0
CLEANED=0
PRESERVE_STATE=0
FORWARD_SPECS=()

select_free_port() {
    python3 - "$1" "${2:-0}" <<'PY'
import socket
import sys
preferred, excluded = map(int, sys.argv[1:])
for port in [preferred, *range(16000, 16100)]:
    if port == excluded:
        continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("No free local port found; set LOCAL_ISAAC_PORT/LOCAL_MJLAB_PORT.")
PY
}

remote_stop() {
    ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" bash -s -- \
        "$RUN_ID" "$CONTAINER" "$ISAAC_PORT" <<'REMOTE_STOP' >/dev/null 2>&1 || true
set -u
run_id=$1
container=$2
isaac_port=$3
base="/tmp/$run_id"
mkdir -p "$base"
touch "$base/cancelled"
stop_group() {
    local file=$1 pid
    if [[ -s "$file" ]]; then
        read -r pid < "$file" || true
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 1
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
        rm -f "$file"
    fi
}
stop_group "$base/proxy.pid"
stop_group "$base/mjlab.pid"
if [[ -e "$base/isaac.attempted" ]]; then
    docker exec "$container" bash -c '
        run_id="$1"; pidfile="/tmp/${run_id}.pid"
        if [[ -s "$pidfile" ]]; then
            read -r pid < "$pidfile" || true
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                kill -TERM -- "-$pid" 2>/dev/null || true
                sleep 1
                kill -KILL -- "-$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile" "/tmp/${run_id}.log"
        fi
    ' -- "$run_id" >/dev/null 2>&1 || true
fi
rm -rf "$base"
REMOTE_STOP
}

cancel_forwards() {
    local spec
    for spec in "${FORWARD_SPECS[@]}"; do
        ssh -T -o BatchMode=yes -o ConnectTimeout=10 -O cancel -L "$spec" "$REMOTE" \
            >/dev/null 2>&1 || true
    done
}

print_ssh_diagnostics() {
    local label=$1
    printf '%s\n' "$label" >&2
    if [[ -s "$STATE_DIR/ssh.out" ]]; then
        printf '%s\n' '--- ssh stdout ---' >&2
        cat "$STATE_DIR/ssh.out" >&2 || true
    fi
    if [[ -s "$STATE_DIR/ssh.err" ]]; then
        printf '%s\n' '--- ssh stderr ---' >&2
        cat "$STATE_DIR/ssh.err" >&2 || true
    fi
}

cleanup() {
    local status=$?
    if (( CLEANED )); then return "$status"; fi
    CLEANED=1
    trap - EXIT INT TERM HUP
    if [[ -n "$SSH_PID" ]] && kill -0 "$SSH_PID" 2>/dev/null; then
        kill -TERM "$SSH_PID" 2>/dev/null || true
        wait "$SSH_PID" 2>/dev/null || true
    fi
    cancel_forwards
    if (( REMOTE_ATTEMPTED )); then
        printf '\nStopping TensorBoards started by this run...\n' >&2
        remote_stop
    fi
    if (( PRESERVE_STATE )); then
        printf 'Local diagnostic files preserved at %s\n' "$STATE_DIR" >&2
    else
        rm -rf "$STATE_DIR"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

preferred=$LOCAL_ISAAC_PORT
LOCAL_ISAAC_PORT=$(select_free_port "$LOCAL_ISAAC_PORT")
if [[ "$preferred" != "$LOCAL_ISAAC_PORT" ]]; then
    printf 'Local port %s is occupied; IsaacLab will use %s. Existing forwards are kept.\n' "$preferred" "$LOCAL_ISAAC_PORT"
fi
preferred=$LOCAL_MJLAB_PORT
LOCAL_MJLAB_PORT=$(select_free_port "$LOCAL_MJLAB_PORT" "$LOCAL_ISAAC_PORT")
if [[ "$preferred" != "$LOCAL_MJLAB_PORT" ]]; then
    printf 'Local port %s is occupied; Mujoco will use %s. Existing forwards are kept.\n' "$preferred" "$LOCAL_MJLAB_PORT"
fi

if ssh -O check "$REMOTE" >/dev/null 2>&1; then
    printf 'Reusing the authenticated SSH connection from your config: %s\n' "$REMOTE"
else
    printf 'No reusable SSH connection. Authenticating with your existing config: %s\n' "$REMOTE"
    ssh -T "$REMOTE" true
fi

# A live ControlMaster socket does not prove that the relay can still
# authenticate a new session.  Probe the actual remote command path without
# prompting; this catches stale relay sessions that exit with status 0 while
# printing an authentication error to stdout.
probe_token="codex_tb_probe_${RANDOM}_$$"
probe_output=''
probe_status=0
probe_output=$(ssh -T -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
    "printf '%s\\n' '$probe_token'" 2>&1) || probe_status=$?
if (( probe_status != 0 )) || ! printf '%s\n' "$probe_output" | tr -d '\r' | grep -Fxq "$probe_token"; then
    printf 'SSH authentication probe failed for %s.\n' "$REMOTE" >&2
    printf '%s\n' "$probe_output" >&2
    printf 'Run `ssh %s` once to refresh the password/OTP session, then rerun this script.\n' "$REMOTE" >&2
    PRESERVE_STATE=1
    exit 1
fi

printf 'Starting TensorBoards and SSH forwarding...\n'
REMOTE_ATTEMPTED=1
FORWARD_SPECS=(
    "127.0.0.1:${LOCAL_ISAAC_PORT}:127.0.0.1:${ISAAC_PORT}"
    "127.0.0.1:${LOCAL_MJLAB_PORT}:127.0.0.1:${MJLAB_PORT}"
)
ssh -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "127.0.0.1:${LOCAL_ISAAC_PORT}:127.0.0.1:${ISAAC_PORT}" \
    -L "127.0.0.1:${LOCAL_MJLAB_PORT}:127.0.0.1:${MJLAB_PORT}" \
    "$REMOTE" bash -s -- "$RUN_ID" "$CONTAINER" "$ISAAC_PORT" "$MJLAB_PORT" "$STARTUP_TIMEOUT" \
    >"$STATE_DIR/ssh.out" 2>"$STATE_DIR/ssh.err" <<'REMOTE_START' &
set -Eeuo pipefail
umask 077
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
run_id=$1
container=$2
isaac_port=$3
mjlab_port=$4
startup_timeout=$5
base="/tmp/$run_id"
mkdir -m 700 "$base"

container_script=$(cat <<'CONTAINER_SCRIPT'
set -eu
mode=$1
run_id=$2
port=$3
pidfile="/tmp/${run_id}.pid"
logfile="/tmp/${run_id}.log"
case "$mode" in
start)
    command -v setsid >/dev/null
    test -x ./isaaclab.sh || { echo "Cannot find ./isaaclab.sh in $PWD" >&2; exit 1; }
    nohup setsid bash -c '
        pidfile=$1; shift
        echo "$$" > "$pidfile"
        exec "$@"
    ' _ "$pidfile" ./isaaclab.sh -p -m tensorboard.main \
        --logdir /workspace/stablemimic/logs/rsl_rl/textop_tracker_g1/ \
        --host 0.0.0.0 --port "$port" >"$logfile" 2>&1 </dev/null &
    for ((i=0; i<100; i++)); do [[ -s "$pidfile" ]] && break; sleep 0.05; done
    test -s "$pidfile"
    ;;
stop)
    if [[ -s "$pidfile" ]]; then
        read -r pid < "$pidfile" || true
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 1
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile" "$logfile"
    fi
    ;;
logs) tail -n 50 "$logfile" 2>/dev/null || true ;;
esac
CONTAINER_SCRIPT
)

stop_group() {
    local file=$1 pid
    if [[ -s "$file" ]]; then
        read -r pid < "$file" || true
        if [[ "$pid" =~ ^[0-9]+$ ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
            sleep 1
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
        rm -f "$file"
    fi
}
stop_services() {
    stop_group "$base/proxy.pid"
    stop_group "$base/mjlab.pid"
    if [[ -e "$base/isaac.attempted" ]]; then
        docker exec "$container" bash -c "$container_script" -- stop "$run_id" "$isaac_port" >/dev/null 2>&1 || true
    fi
}
on_exit() {
    local status=$?
    trap - EXIT INT TERM HUP
    if (( status != 0 )); then
        printf '\nRemote startup failed. Logs: %s\n' "$base" >&2
        tail -n 50 "$base/mjlab.log" >&2 2>/dev/null || true
        tail -n 50 "$base/proxy.log" >&2 2>/dev/null || true
        docker exec "$container" bash -c "$container_script" -- logs "$run_id" "$isaac_port" >&2 || true
        stop_services
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
check_cancelled() { [[ ! -e "$base/cancelled" ]] || { echo 'Startup cancelled.' >&2; exit 130; }; }
check_cancelled
command -v python3 >/dev/null
command -v setsid >/dev/null
command -v uv >/dev/null || { echo 'Cannot find uv on deploy.' >&2; exit 1; }
test -d "$HOME/ws/mujoco_lab/mjlab" || { echo "Cannot find $HOME/ws/mujoco_lab/mjlab" >&2; exit 1; }
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]]; then
    echo "Container $container is stopped. Start it before running this script." >&2
    exit 1
fi
container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{if .IPAddress}}{{.IPAddress}} {{end}}{{end}}' "$container" | awk '{print $1}')
test -n "$container_ip" || { echo 'Container has no IPv4 address.' >&2; exit 1; }

python3 - "$container_ip" "$isaac_port" "$mjlab_port" <<'PORT_CHECK'
import socket
import sys
container_ip, isaac, mjlab = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for port in (isaac, mjlab):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(('127.0.0.1', port))
        except OSError as exc: raise SystemExit(f'Remote port {port} is occupied: {exc}')
try:
    conn = socket.create_connection((container_ip, isaac), timeout=2)
except ConnectionRefusedError:
    pass
except OSError as exc:
    raise SystemExit(f'Cannot reach container {container_ip}: {exc}')
else:
    conn.close()
    raise SystemExit(f'Container port {isaac} is occupied: stop the old IsaacLab TensorBoard first.')
PORT_CHECK

touch "$base/isaac.attempted"
docker exec "$container" bash -c "$container_script" -- start "$run_id" "$isaac_port"
check_cancelled
(
    cd "$HOME/ws/mujoco_lab/mjlab"
    nohup setsid bash -c '
        pidfile=$1; shift
        echo "$$" > "$pidfile"
        exec "$@"
    ' _ "$base/mjlab.pid" uv run --no-sync tensorboard \
        --logdir logs/rsl_rl/ --host 127.0.0.1 --port "$mjlab_port" \
        >"$base/mjlab.log" 2>&1 </dev/null &
)
for ((i=0; i<100; i++)); do [[ -s "$base/mjlab.pid" ]] && break; sleep 0.05; done
test -s "$base/mjlab.pid"
check_cancelled

cat > "$base/proxy.py" <<'PROXY'
import socket
import socketserver
import sys
import threading

target = (sys.argv[1], int(sys.argv[2]))

def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data: break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try: dst.shutdown(socket.SHUT_WR)
        except OSError: pass

class Proxy(socketserver.BaseRequestHandler):
    def handle(self):
        try: upstream = socket.create_connection(target, timeout=10)
        except OSError: return
        with upstream:
            threading.Thread(target=pipe, args=(self.request, upstream), daemon=True).start()
            pipe(upstream, self.request)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

with Server(('127.0.0.1', int(sys.argv[2])), Proxy) as server:
    server.serve_forever()
PROXY
nohup setsid bash -c '
    pidfile=$1; shift
    echo "$$" > "$pidfile"
    exec "$@"
' _ "$base/proxy.pid" python3 "$base/proxy.py" "$container_ip" "$isaac_port" \
    >"$base/proxy.log" 2>&1 </dev/null &
for ((i=0; i<100; i++)); do [[ -s "$base/proxy.pid" ]] && break; sleep 0.05; done
test -s "$base/proxy.pid"
check_cancelled

printf 'Waiting for both remote TensorBoards (up to %ss)...\n' "$startup_timeout"
python3 - "$isaac_port" "$mjlab_port" "$startup_timeout" "$base/cancelled" <<'READY'
from pathlib import Path
import sys, time, urllib.error, urllib.request
isaac, mjlab, timeout, cancelled = sys.argv[1:]
pending = {f'http://127.0.0.1:{isaac}/', f'http://127.0.0.1:{mjlab}/'}
deadline = time.monotonic() + int(timeout)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
while pending and time.monotonic() < deadline:
    if Path(cancelled).exists(): raise SystemExit('Startup cancelled.')
    for url in tuple(pending):
        try:
            with opener.open(url, timeout=1) as response:
                if response.status == 200: pending.remove(url)
        except (OSError, urllib.error.URLError):
            pass
    if pending: time.sleep(0.5)
if pending: raise SystemExit('TensorBoard did not become ready: ' + ', '.join(sorted(pending)))
READY
check_cancelled
printf 'READY CONTAINER_IP=%s\n' "$container_ip"
# Keep this session alive while its two forwarding listeners are in use.
while :; do sleep 3600 & wait $!; done
REMOTE_START
SSH_PID=$!

# Wait until the remote readiness check has completed.
deadline=$((SECONDS + STARTUP_TIMEOUT + 15))
while (( SECONDS < deadline )); do
    if grep -q '^READY ' "$STATE_DIR/ssh.out" 2>/dev/null; then break; fi
    if ! kill -0 "$SSH_PID" 2>/dev/null; then
        print_ssh_diagnostics 'SSH session exited during startup.'
        PRESERVE_STATE=1
        exit 1
    fi
    sleep 1
done
if ! grep -q '^READY ' "$STATE_DIR/ssh.out" 2>/dev/null; then
    print_ssh_diagnostics 'Timed out waiting for remote TensorBoard startup.'
    PRESERVE_STATE=1
    exit 1
fi

# Check the actual local forwarding endpoints. Report a useful error instead
# of exposing a Python traceback when a relay rejects a direct connection.
if ! python3 - "$LOCAL_ISAAC_PORT" "$LOCAL_MJLAB_PORT" <<'LOCAL_CHECK'
import sys, urllib.error, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in sys.argv[1:]:
    try:
        with opener.open(f'http://127.0.0.1:{port}/', timeout=5) as response:
            if response.status != 200: raise SystemExit(f'HTTP {response.status}')
    except (OSError, urllib.error.URLError) as exc:
        raise SystemExit(f'http://127.0.0.1:{port}/ is unavailable: {exc}')
LOCAL_CHECK
then
    print_ssh_diagnostics 'The remote services started, but local SSH forwarding is unavailable.'
    echo 'The remote services started, but local SSH forwarding is unavailable.' >&2
    PRESERVE_STATE=1
    echo "Try setting LOCAL_ISAAC_PORT/LOCAL_MJLAB_PORT explicitly; inspect: $STATE_DIR" >&2
    exit 1
fi

printf '\nTensorBoard is ready:\n'
printf '  IsaacLab: http://127.0.0.1:%s\n' "$LOCAL_ISAAC_PORT"
printf '  Mujoco:   http://127.0.0.1:%s\n' "$LOCAL_MJLAB_PORT"
printf 'Press Ctrl+C to stop this run; your shared SSH connection stays open.\n'
if [[ "$OPEN_BROWSER" == 1 ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:$LOCAL_ISAAC_PORT" >/dev/null 2>&1 &
        xdg-open "http://127.0.0.1:$LOCAL_MJLAB_PORT" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        open "http://127.0.0.1:$LOCAL_ISAAC_PORT" "http://127.0.0.1:$LOCAL_MJLAB_PORT" >/dev/null 2>&1 || true
    fi
fi
wait "$SSH_PID"
