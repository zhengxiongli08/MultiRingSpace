
import sys
import os

class Logger():
    def __init__(self, result_path="./result", log_name="result.log"):
        log_path = os.path.join(result_path, log_name)
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=64, encoding="utf-8")
        
        return
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
        return
 
    def print(self, *message):
        message = ",".join([str(it) for it in message])
        self.terminal.write(str(message) + "\n")
        self.log.write(str(message) + "\n")
        self.flush()
        
        return

    def close(self):
        self.log.close()

        return
        
if __name__ == "__main__":
    from time import sleep
    logger = Logger()
    for i in range(10):
        logger.print("Trying!")
        sleep(0.5)
