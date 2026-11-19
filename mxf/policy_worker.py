"""Run the frozen LIBERO policy with a pause-tolerant private local transport."""

from functools import partial

from local_platform.vla import libero_metric_worker


def main():
    # The evaluator owns this process, closes the private socket on exit, and
    # enforces its own startup and in-flight RPC deadlines. An idle evaluator
    # (including a deliberately suspended simulation) consumes no policy draw.
    serve = libero_metric_worker.serve
    try:
        libero_metric_worker.serve = partial(serve, timeout_seconds=None)
        libero_metric_worker.main()
    finally:
        libero_metric_worker.serve = serve


if __name__ == "__main__":
    main()
