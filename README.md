# DistTrainBench

DistTrainBench is an early-stage benchmark prototype for evaluating an AI agent's ability to design and run distributed model training under constrained container resources.

## Benchmark concept

The agent runs in a coordinator container and plans and coordinates training across worker containers connected through a Docker bridge network. The prototype gives the coordinator and workers different CPU and memory limits and rate-limits inter-container traffic.

The training task is to train ResNet-20 on CIFAR-100. The agent runs for a fixed wall-clock budget (one hour by default), then the harness scores its final checkpoint using top-1 accuracy on the CIFAR-100 test split.

The scored architecture is `disttrainbench.ResNet20`: a 3x3 CIFAR stem, three stages with three basic blocks each, 16/32/64 channels, 3x3 convolutional projection shortcuts, global 8x8 average pooling, and a 100-class output layer. The model implementation is included in the container image so agents can import the exact architecture.

### Data and integrity rules

- Downloading CIFAR-100 is allowed, including a dataset archive that contains the test split.
- The test split must not be used for training or evaluated before the final evaluation.
- Downloading or using pretrained model weights is not allowed.
- Agents may still attempt prohibited actions. Run traces should be reviewed to identify these violations, following the observational approach used by PostTrainBench.

## Current prototype

The current Docker setup starts one coordinator and eight workers from the same CPU-only image on a per-run Docker bridge network. The workers are reachable as `worker-1` through `worker-8`. A manual Compose start uses `/workspace`, `/data`, and `/runs`; the evaluation harness mounts separate host directories for each run.

The coordinator defaults to 0.125 CPU and 4 GB of memory. Each of the eight workers defaults to 0.125 CPU and 1 GB of memory. The combined defaults are 1.125 CPUs and 12 GB across all nine training containers. Workers have identical limits, and the coordinator has the same CPU limit to constrain training done there while retaining memory for the agent. Inter-container traffic is capped at 200 kbit/s aggregate in each direction per container, from `1 Gbps / (5 * 10^3)`. Egress uses a token-bucket shaper; ingress uses policing that drops excess packets. At startup, each training container gets a fixed egress delay sampled uniformly as an integer from 1 to 700 ms. A round trip includes the delays set on both endpoints. The selected delays appear in the agent prompt and run metadata. External traffic, including CIFAR downloads and agent API calls, is not rate-limited. No additional random delay or random packet loss is configured. One-shot shaper sidecars apply Linux `tc` in each container's network namespace; training containers drop `NET_ADMIN` so the agent cannot remove the configured qdisc.

The agent image includes the Codex CLI and Claude Code. Provider keys are read from `.env` by the host runner and injected only into the selected CLI process in the coordinator; the coordinator and workers do not inherit them as container-wide environment variables. Workers receive only the per-run worker API token.

The image includes Python 3.12, CPU-only PyTorch 2.15.0 with torchvision 0.29.2, Pillow, NumPy, Requests, bash, curl, git, Node.js 22, Codex CLI 0.144.0, and Claude Code 2.1.219. Other Python packages may be installed by the agent.

## Run an evaluation

The harness has built-in non-interactive launchers for Codex CLI and Claude Code. Put the matching key in `.env`, then run one of:

```bash
python3 scripts/evaluate_run.py \
  --agent codex \
  --time-limit-seconds 3600

python3 scripts/evaluate_run.py \
  --agent claude \
  --time-limit-seconds 3600
```

Add `--model-name MODEL_ID` to pin an explicit model supported by the chosen CLI; `--model` is an alias. If neither is set, the CLI's configured default is used. The selected identifier is recorded in run metadata.

The default task prompt is saved as `runs/<run-id>/agent.prompt.txt` and sent to the CLI on standard input. Codex runs in JSONL event mode; Claude Code runs in stream-JSON mode. Their output is saved in `agent.stdout.log` and `agent.stderr.log` for trace review, with configured keys and tokens redacted before publication. These modes expose tool and command activity, which can help reviewers identify use of the test split or pretrained weights; trace review remains observational and does not prevent prohibited actions. Override the task prompt with `--prompt-file PATH`.

The harness snapshots `workspace/`, creates run-specific data and output directories, builds the image, and starts one coordinator and the configured workers. The agent runs in the coordinator until it exits or reaches the time limit. It sets these environment variables for the agent:

- `DISTTRAINBENCH_RUN_ID` and `DISTTRAINBENCH_RUN_DIR` (set to `/runs`)
- `DISTTRAINBENCH_CHECKPOINT`, the required final checkpoint path
- `DISTTRAINBENCH_TIME_LIMIT_SECONDS`
- `DISTTRAINBENCH_TRAIN_DATA_DIR`, set to `/data`
- `DISTTRAINBENCH_WORKERS`, a comma-separated list of worker service names

For Codex, set `CODEX_API_KEY` in `.env`; for Claude Code, set `ANTHROPIC_API_KEY`. The selected key is provided only to that CLI invocation. For Codex, `OPENAI_API_KEY` is accepted as a fallback for existing local setups. For a custom launcher, use `--agent-command '...'`; it receives no provider key by default, and `--api-key-env CODEX_API_KEY` or `--api-key-env ANTHROPIC_API_KEY` explicitly selects one. Custom launchers can receive a prompt on stdin with `--prompt-file PATH`.

The agent must save a model `state_dict` to `DISTTRAINBENCH_CHECKPOINT` (`/runs/checkpoint.pt`). It may also save a dictionary containing a `state_dict` entry. Import the scored model with `from disttrainbench import ResNet20`. Use CIFAR-100's training split for training; `torchvision.datasets.CIFAR100(root='/data', train=True, download=True, ...)` is a typical loader. Its archive may include the test split, as allowed by the data rules.

Docker network connectivity lets the coordinator reach workers by service name; it does not execute commands by itself. Each worker runs an authenticated HTTP job API on port 8765, reachable only inside that run's bridge network. The coordinator gets `DISTTRAINBENCH_WORKER_TOKEN`. Submit a process with `POST http://worker-1:8765/jobs` and a JSON body such as `{"command":["python","/workspace/train_worker.py"],"cwd":"/workspace"}` plus `Authorization: Bearer $DISTTRAINBENCH_WORKER_TOKEN`. Poll `GET /jobs/<job_id>` for status and `GET /jobs/<job_id>/output` for recent output. Use `DELETE /jobs/<job_id>` to stop a job. The worker API records commands, lifecycle, and process output in Docker logs, which the harness saves as `container.log`. The port is not published on the host.

Each run gets its own workspace snapshot, empty training-data directory, and output directory. It cannot read files from other run output directories. After the agent exits, the harness stops and removes the training containers, then starts a separate evaluator container to download/load the test split and score the checkpoint. The evaluator's data directory is not mounted into the coordinator or workers during training. This keeps the harness from exposing the test data, but it cannot stop an agent from independently downloading the allowed archive and opening its test files. Review the saved agent output and any agent-specific event trace for prohibited use. Harness metadata and logs are held outside the mounted directories while the agent runs, then published after the containers stop.

Each run is saved under `runs/<run-id>/`:

- `metadata.json`: host and container limits, network profile and rate, software versions, measured host throughput, duration, and score
- `events.jsonl`: timestamped run lifecycle events
- `agent.prompt.txt`: the task prompt provided to the agent, when one is used
- `agent.stdout.log` and `agent.stderr.log`: CLI JSON/event stream and stderr, with secret values redacted
- `container.log`: Docker Compose logs
- `evaluation.json` and `evaluation.log`: final test result and evaluator output
- `agent-workspace/`: the post-run snapshot of the agent's working files

Before publishing artifacts, the harness redacts configured API keys and tokens from text files, including traces, container logs, and agent-created workspace files. Trace hashes in `metadata.json` refer to the redacted log files.

The host FP32 benchmark uses OpenBLAS SGEMM with a 4096x4096 matrix and up to two threads by default. Its library version/configuration, actual thread count, and measured throughput are recorded. If OpenBLAS is not installed on the host, the metadata records that the measurement was unavailable. The run also records host CPU details, worker count, each container's CPU and memory limits, network profile, PyTorch and torchvision versions, Codex CLI and Claude Code versions, agent model and duration, and final top-1 score. Throughput is recorded as run context; it is not used to rescale the score.

Use the same time limit and resource settings when comparing runs. `NETWORK_RATE_BPS` in `.env` sets the inter-container cap (200000 by default); `--network-rate-bps` overrides it for one run. The rate is expressed in bits per second. The Docker host kernel must support `tc` ingress policing, TBF egress shaping, and `netem` delay. `--no-build` reuses the existing image, `--skip-host-benchmark` skips the host SGEMM measurement, and `--run-id` sets a readable run directory name.

## Start the containers

```bash
cp .env.example .env
docker compose up --build -d
```

Adjust `COORDINATOR_CPUS`, `COORDINATOR_MEMORY`, `WORKER_CPUS`, and `WORKER_MEMORY` in `.env` after checking the host's available resources. All eight workers use the same `WORKER_*` limits. The default limits across the coordinator and workers add up to 1.125 CPUs and 12 GB of memory.

Useful commands:

```bash
docker compose exec coordinator bash
docker compose exec worker-1 bash
docker compose exec worker-8 bash
docker compose down
```
