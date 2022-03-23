import logging

logging.basicConfig(filename="mnist-digit-classification.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
        
log = logging.getLogger()
 
log.setLevel(logging.DEBUG)