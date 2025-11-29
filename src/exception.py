import os
import sys
from src.logger import logging

def display_error_message(error,error_detail:sys):
    _,_,exc_tab=error_detail.exc_info()
    file_name=exc_tab.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tab.tb_lineno,str(error))

    return error_message
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_msg=display_error_message(error_message,error_detail)
        logging.error(self.error_msg)

    def __str__(self):
        return self.error_msg