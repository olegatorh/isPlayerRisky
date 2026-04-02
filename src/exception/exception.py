class RiskyException(Exception):
    def __init__(self, error_message, error_details):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (f"\n\nerror occurred in python script name [{self.filename}] line number [{self.lineno}]"
                f"error message [{str(self.error_message)}]\n\n")
