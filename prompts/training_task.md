# Training task

Train the fixed DistTrainBench ResNet-20 model on CIFAR-100's training split. Maximize its final top-1 accuracy within the wall-clock budget. Plan and coordinate work across the available worker containers through their HTTP job APIs.

## Required result

- Save the final model's `state_dict` to `{{CHECKPOINT}}` before the deadline. The evaluator instantiates `disttrainbench.ResNet20` and loads this checkpoint strictly.
- Import the exact scored model with `from disttrainbench import ResNet20`.
- The model has a 3x3 CIFAR stem, three stages of three basic blocks, 16/32/64 channels, 3x3 projection shortcuts, 8x8 average pooling, and 100 output classes.
- The training data directory is `{{DATA_DIR}}`. Download CIFAR-100 if needed; downloading the dataset archive, including its test files, is allowed.

## Data and integrity rules

- Use only CIFAR-100's training split for training, validation, tuning, and checkpoint selection.
- Do not inspect, evaluate on, or train from the CIFAR-100 test split before your run ends. The archive may contain test files, but their presence does not permit using them.
- Do not download or use pretrained weights.
- The final test evaluation runs in a separate evaluator after your process exits. Its result is not available during training.

## Distributed workers

Available workers: `{{WORKERS}}`. They share `/workspace`, `/data`, and `/runs` with the coordinator. Each worker exposes an authenticated HTTP API on port 8765 inside the Docker network. The per-run bearer token is in `DISTTRAINBENCH_WORKER_TOKEN`.

Submit a worker process with a JSON POST to `http://WORKER:8765/jobs`, an `Authorization: Bearer $DISTTRAINBENCH_WORKER_TOKEN` header, and a JSON body such as:

```json
{"command":["python","/workspace/train_worker.py"],"cwd":"/workspace"}
```

Poll `GET http://WORKER:8765/jobs/JOB_ID` for status and `GET http://WORKER:8765/jobs/JOB_ID/output` for output. Send `DELETE` to `/jobs/JOB_ID` to stop a job. Worker commands run inside the corresponding worker container. The coordinator can also run local work. Choose a useful division of training and combine the results into one final checkpoint.

## Runtime information

- Run ID: `{{RUN_ID}}`
- Wall-clock budget: `{{TIME_LIMIT}}` seconds
- Coordinator checkpoint path: `{{CHECKPOINT}}`
- Training data path: `{{DATA_DIR}}`
- Worker names: `{{WORKERS}}`
- Workers may have different compute, memory, and network limits. Discover the available resources and design accordingly.

Do not rely on access to the host Docker daemon. Use the worker HTTP APIs for remote process execution. Leave enough time to write and verify the final checkpoint before the deadline.
