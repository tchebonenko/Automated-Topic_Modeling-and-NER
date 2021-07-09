import os

# replace with environment variables whereever possible
DEBUG = True

SECRET_KEY = '3555568jj0b13ok0c676dfde853gh789'


UPLOADED_PATH = os.path.abspath(os.path.dirname(__file__))
# muximum upload size 30Mb otherwise page with 413 status code
MAX_CONTENT_LENGTH = 1024 * 1024 * 30
UPLOAD_EXTENSIONS = ['.csv']
