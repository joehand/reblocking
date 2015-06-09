import os

from flask import Flask, render_template

from .frontend import frontend

from .config import Config, DevelopmentConfig, ProductionConfig
from .extensions import assets

# For import *
__all__ = ['create_app']

DEFAULT_BLUEPRINTS = (
    frontend,
)

def create_app(config=None, app_name=None, blueprints=None):
    '''Create a Flask app.'''

    if app_name is None:
        app_name = Config.PROJECT
    if blueprints is None:
        blueprints = DEFAULT_BLUEPRINTS

    app = Flask(app_name, instance_relative_config=True)
    configure_app(app, config)
    configure_hook(app)
    configure_blueprints(app, blueprints)
    configure_extensions(app)
    configure_logging(app)
    configure_template_filters(app)
    configure_error_handlers(app)

    return app

def configure_app(app, config=None):
    '''Different ways of configurations.'''

    # http://flask.pocoo.org/docs/api/#configuration
    if config:
        app.config.from_object(config)
    else:
        try:
            from local import LocalConfig
            app.config.from_object(LocalConfig)
        except:
            app.config.from_object(DevelopmentConfig)

def configure_extensions(app):
    # Flask assets
    assets.init_app(app)


def configure_blueprints(app, blueprints):
    '''Configure blueprints in views.'''

    for blueprint in blueprints:
        app.register_blueprint(blueprint)

def configure_template_filters(app):
    pass

def configure_logging(app):
    '''Configure file(info) and email(error) logging.'''

    if app.debug or app.testing:
        #skip loggin
        return

def configure_hook(app):

    @app.before_request
    def before_request():
        pass

def configure_error_handlers(app):

    @app.errorhandler(403)
    def forbidden_page(error):
        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def server_error_page(error):
        return render_template('errors/500.html'), 500
