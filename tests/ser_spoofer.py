class Fake_ser():
    def __init__(self):
        self.is_open = -1
        self.port = -1
        self.baudrate = -1
        self.rtscts = -1
        self.in_waiting = 0
    def open(self):
        self.is_open = True
    def flush(self):
        print("wash, flush, clean!")
    def flushInput(self):
        print("wash, flush, clean!")
    def flushOutput(self):
        print("wash, flush, clean!")
    def close(self):
        self.is_open = False
    def write(self,b):
        print("Call Me Ishmale,"+b)
        return b
    def read_until(self,size=9999):
        return "F"*size
class list_ports():
    def __init__(self,on = True):
        self.port = "VID:PID=0403:6015"
        if not on:
            self.port = "gaga"
        print("made")
    def comports(self):
        ports  = [self.port,self.port,self.port]
        return ports
