from flask import Blueprint
from app.controllers.Controller import Controller

def Routes(app):
    controller = Controller()
    controller_bp = Blueprint('controller', __name__)
    controller_bp.route('/', methods=['POST'])(controller.ProcessImages)
    app.register_blueprint(controller_bp)
