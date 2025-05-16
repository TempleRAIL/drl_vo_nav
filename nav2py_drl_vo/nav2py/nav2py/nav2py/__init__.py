import argparse
import sys
import typing
from .interfaces import nav2py_costmap_controller


def main(cls: typing.Type[nav2py_costmap_controller]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()

    controller = cls(
        host=args.host,
        port=args.port,
    )
    try:
        controller.serve()
    except KeyboardInterrupt:
        controller.cleanup()
