from sys import stdin
from select import select
import tty, termios


class GetCh():
    # also messes up output; maybe something more sophisticated needs to be done
    def __enter__(self):
        self.fd = stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setraw(self.fd, termios.TCSANOW)
        return self
    def getch(self):
        i, o, e = select([self.fd], [], [], 0)
        if i:
            ch = stdin.read(1)
        else:
            ch = ''
        return ch
    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_settings)

def getch():
    i, o, e = select([stdin], [], [], 0)
    if i:
        ch = stdin.read(1)
    else:
        ch = ''
    return ch
