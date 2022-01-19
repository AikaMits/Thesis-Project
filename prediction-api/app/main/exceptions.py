class NotFoundException(Exception):
    msg = 'Not found'

    def __init__(self,msg):
        self.msg = msg