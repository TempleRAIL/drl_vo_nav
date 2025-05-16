"""
nav2py interfaces
"""

import abc
import socket
import struct
import sys
import typing


class nav2py_costmap_controller(abc.ABC):
    """
    nav2py costmap controller base
    """

    _socket: socket.socket
    _conn: socket.socket
    __callbacks: typing.Dict[str, typing.Callable]

    def _register_callback(self, name: str, fn: typing.Callable):
        self.__callbacks[name] = fn

    def __init__(
        self,
        host: str,
        port: int
    ) -> None:

        self.__callbacks = {}

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((host, port))
        self._socket.listen()

        port = self._socket.getsockname()[1]
        port_message = struct.pack('!H', port)
        sys.stdout.buffer.write(port_message)
        sys.stdout.buffer.write(int.to_bytes(187, 1, 'big'))
        sys.stdout.buffer.write(int.to_bytes(201, 1, 'big'))
        sys.stdout.buffer.flush()

    def serve(self):
        conn, addr = self._socket.accept()
        self._conn = conn

        SEP_BYTE = int.to_bytes(1, 1, 'big')
        END_BYTE = int.to_bytes(3, 1, 'big')

        def receive_msg() -> typing.Tuple[str, typing.List[bytes]]:
            buffer: bytes = bytes()
            while True:
                byte = conn.recv(1)
                if byte == END_BYTE:
                    break
                buffer += byte

            name, *content = buffer.split(SEP_BYTE)
            return name.decode(), content

        with conn:
            while True:
                name, content = receive_msg()
                if name in self.__callbacks:
                    self.__callbacks[name](content)

    def _send_cmd_vel(self, linear_x: float, angular_z: float):
        self._conn.send(struct.pack('!d', linear_x))
        self._conn.send(struct.pack('!d', angular_z))

    def cleanup(self):
        self._socket.close()
