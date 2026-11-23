import socket
import threading
import time

from drone_control.stats import Stats


class Tello:
    def __init__(self):
        self.local_ip = ''
        self.local_port = 8889
        self.tello_ip = '192.168.10.1'
        self.tello_port = 8889
        self.tello_adderss = (self.tello_ip, self.tello_port)
        self.log = []
        self.MAX_TIME_OUT = 15.0
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((self.local_ip, self.local_port))
        except OSError:
            self.socket.close()
            raise
        self._lock = threading.Lock()
        self._reply_unreliable = False

    def send_command(self, command, timeout=None):
        """Send one SDK command and return its reply, or None on timeout."""
        with self._lock:
            stat = Stats(command, len(self.log))
            self.log.append(stat)
            print(f"sending command: {command}")
            self.socket.sendto(command.encode('utf-8'), self.tello_adderss)
            if self._reply_unreliable:
                # a late reply cannot be matched reliably after a timeout
                return None
            deadline = time.monotonic() + (self.MAX_TIME_OUT if timeout is None else timeout)
            try:
                while time.monotonic() < deadline:
                    self.socket.settimeout(max(0.001, deadline - time.monotonic()))
                    reply, address = self.socket.recvfrom(1024)
                    if address == self.tello_adderss:
                        stat.add_response(reply)
                        return reply.decode('utf-8', errors='replace').strip()
            except socket.timeout:
                pass
            except KeyboardInterrupt:
                self._reply_unreliable = True
                raise
            self._reply_unreliable = True
            return None

    def on_close(self):
        self.socket.close()

    def get_log(self):
        return self.log
