# Quick Screen + Dev Container Setup

## Copy-Paste Commands

### 1. Start persistent screen session

```bash
screen -S dev
```

### 2. Inside screen - Build Docker image

```bash
# Make sure you are in the irange directory on your host
# cd ~/irange
docker build -f .devcontainer/Dockerfile -t irange-dev .
```

### 3. Inside screen - Run container

```bash
# This mounts your current directory to /workspaces/irange inside the container
docker run -it --gpus all -v $(pwd):/workspaces/irange irange-dev
```

### 4. Now you're inside the container terminal (inside screen)

```bash
cd /workspaces/irange
# Ready to run your scripts
```

### 5. Detach screen (leave everything running)

```
Ctrl+A, then D
```

### 6. Later - Reattach to check progress

```bash
screen -r dev
```

---

## That's it!

Now inside the container terminal you can run:
```bash
./scripts/setup_folders.sh
./scripts/make_pq.sh all
./scripts/run_tests.sh
./scripts/execute_experiments.sh all
```

---

## Scroll in Screen

### To scroll up/down in screen history:

1. Press and hold: **Ctrl+A** (together)
2. Release both keys
3. Press: **[** (then release)

Then use:
- **Arrow keys** or **Page Up/Page Down** to scroll
- **q** to exit scroll mode

### Example:

While attached to screen, to see output that scrolled off:

```
Ctrl+A          (press and hold, then release)
[               (press this key)
↑ ↑ ↑           (scroll up with arrow keys)
q               (press to exit scroll mode)
```
