#!/usr/bin/env python
# coding: utf-8
import signal

from cli import CLI


def stop_handler(signum, frame):
    _ = (signum, frame)
    exit(1)


def main():
    """model = SimplexCentroid(
        n_points=4,
        lower_bounds=(0, 0, 0, 0.4),
        upper_bounds=None,
        test_points=(
            (0.12, 0.12, 0.33, 0.43),
            (0.08, 0.12, 0.20, 0.60),
            (0.03, 0.22, 0.06, 0.69),
        ),
    )
    print(model.__repr__())"""
    signal.signal(signalnum=signal.SIGINT, handler=stop_handler)
    cli = CLI()
    cli.loop()


if __name__ == "__main__":
    main()
