from flask import Flask, make_response
from flask_cors import CORS

from app.main.api_v1 import api_v1
from app.main.exceptions import NotFoundException

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.register_blueprint(api_v1)


@app.errorhandler(NotFoundException)
@app.errorhandler(404)
def not_found(error):
    msg = error.msg if hasattr(error, 'msg') else 'Not found'
    return make_response('{"error": "%s"}' % msg, 404)

