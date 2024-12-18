import logging         
import sys   
import os                        
from datetime import datetime                            
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)  

#Use this Log while development
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"                    
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)                                                
try:
    os.makedirs(logs_path, exist_ok=False)
except FileExistsError:
    pass                                                  

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)                         

logging.basicConfig(                                                                          
    filename=LOG_FILE_PATH,                                      
    format= "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",    
    level= logging.INFO    
)

