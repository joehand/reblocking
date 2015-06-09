import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):

    PROJECT = 'reblocker'

    # Get app root path
    PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    APP_ROOT = os.path.abspath(os.path.dirname(__file__))

    DEBUG = True
    ASSETS_DEBUG = True

    SECRET_KEY = 'this_is_so_secret' #used for development, reset in prod

    PRODUCTION = False

    UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')

class ProductionConfig(Config):

    PRODUCTION = True

    SECRET_KEY = os.environ.get('SECRET_KEY')

    SECURITY_PASSWORD_SALT = os.environ.get('SECURITY_PASSWORD_SALT')

    DEBUG = False

    ASSETS_AUTO_BUILD = False

class DevelopmentConfig(Config):

    SECURITY_PASSWORD_SALT = '/2aX16zPnnIgfMwkOjGX4S'

class TestingConfig(Config):

    TESTING = True
