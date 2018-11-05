import threading

class PointlessExampleThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.return_value = None
        
    def run(self):
        print 'Hello from the thread you just spawned!'
        self.return_value = 123
        
    def join(self):
        threading.Thread.join(self)
        return self.return_value
    

NewThread = PointlessExampleThread()
NewThread.start()
thread_output = NewThread.join()
print 'The thread completed and returned this value: %s' % thread_output
