# -*- coding: utf-8 -*-  

"""
Created on 2021/07/15

@author: Ruoyu Chen
"""

class Logger():
    """
    Logging the process
    """
    def __init__(self, logging_path):
        super(Logger, self).__init__()
        self.logging_path = logging_path
        self.replace_str = ["\033[0m","\033[1m","\033[4m","\033[5m","\033[7m","\033[8m",
                            "\033[30m","\033[31m","\033[32m","\033[33m","\033[34m","\033[35m","\033[36m","\033[37m"]
    
    def write(self, txt):
        print(txt)
        for rep in self.replace_str:
            txt = txt.replace(rep,"")
        with open(self.logging_path,"a") as file:
            file.write(txt+'\n')
        
    