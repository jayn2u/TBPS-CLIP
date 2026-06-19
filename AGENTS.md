- Use `uv run python` to execute Python code.
## Dataset location

Lab datasets are stored at one of:

- `/mnt/data/lab_datasets`
- `/data/jayn2u/lab_datasets`

These paths refer to the same storage. Use whichever exists on the current machine.

Set `anno_dir` and `image_dir` in `config/config.yaml` (or `config/s.config.yaml`) to paths under the chosen root. For CUHK-PEDES, annotations are in the dataset directory and images are under `{root}/CUHK-PEDES/imgs`.
