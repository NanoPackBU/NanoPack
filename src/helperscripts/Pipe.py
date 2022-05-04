import win32file
import win32pipe


class PipeServer:
    def __init__(self, pipeName):
        self.pipe = win32pipe.CreateNamedPipe(
            r'\\.\pipe\\' + pipeName,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            win32pipe.PIPE_UNLIMITED_INSTANCES,
            65536, 65536, 2000, None)

    # Careful, this blocks until a connection is established
    def connect(self):
        win32pipe.ConnectNamedPipe(self.pipe, None)

    # Message without tailing '\n'
    def write(self, message):
        win32file.WriteFile(self.pipe, message.encode() + b'\n')

    def read(self):
        bufSize = 4096
        win32file.SetFilePointer(self.pipe, 0, win32file.FILE_BEGIN)
        result, data = win32file.ReadFile(self.pipe, bufSize, None)
        buf = data
        while len(data) == bufSize:
            result, data = win32file.ReadFile(self.pipe, bufSize, None)
            buf += data
        return buf.decode('utf8').split('\r\n')

    def close(self):
        return win32file.CloseHandle(self.pipe)
