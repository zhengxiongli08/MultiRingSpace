
import sys

LOG_PATH = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/result/result.log"

class Logger():
    def __init__(self, log_path=LOG_PATH):
        
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
