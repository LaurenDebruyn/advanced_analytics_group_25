from flask import Flask
# from werkzeug.middleware.proxy_fix import ProxyFix

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)

# app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
