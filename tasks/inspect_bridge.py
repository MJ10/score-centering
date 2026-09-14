from pathlib import Path


def eval_inspect(model, make_tasks, log_dir, seq_len=1024, limit=1024):
    """Run inspect-ai tasks (make_tasks() -> list) against the local server;
    returns flat inspect/<task>/<metric> scores and copies the logs out."""
    import shutil
    import tempfile

    import httpx
    from inspect_ai import eval_set
    from inspect_ai.model import get_model

    from models import server

    port = server.start(model, seq_len=seq_len)
    try:
        http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=None), timeout=600)
        inspect_model = get_model(
            "openai/custom",
            base_url=f"http://localhost:{port}/v1",
            api_key="key",
            http_client=http_client,
        )
        with tempfile.TemporaryDirectory() as tmp_log_dir:
            _, logs = eval_set(
                make_tasks(),
                model=inspect_model,
                max_connections=2048,
                limit=limit,
                log_dir=tmp_log_dir,
            )
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp_log_dir, str(log_dir), dirs_exist_ok=True)
        results = {}
        for log in logs:
            task_name = log.eval.task
            for k, v in log.results.scores[0].metrics.items():
                results[f"inspect/{task_name}/{k}"] = v.value
        return results
    finally:
        server.stop()
