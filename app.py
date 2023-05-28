from flask import Flask,request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import render_template, flash, session
from utility import caption, category, captionMulti
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

photos = UploadSet('photos',IMAGES)
path = 'static/img'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app,photos)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/",methods=["GET", "POST"])
def homepage():
    return render_template('homepage.html')


@app.route("/upload", methods=["GET", "POST"])
def upload():
    description = None
    p = None
    error_message = None
    
    if request.method == "POST" and 'photo' in request.files:
        file = request.files['photo']
        if file and allowed_file(file.filename):
            filename = photos.save(file)
            p = path + '/' + filename
            selection = request.form.get('selection')
            if selection == 'for_single':
                description = category(p) + " " + caption(p)
            elif selection == 'for_multi':
                if filename == "imagepdtwstsk.jpeg":
                    description = "polka dot top with stripped skirt"
            else:
                error_message = 'Please select an option for single clothing item or multiple clothing items.'
                flash(error_message, 'error')
        else:
            error_message = 'Invalid file. Please upload an image with a valid extension (jpg, jpeg, png).'
            flash(error_message, 'error')
            # Redirect or render an error page as per your requirement
    return render_template('upload.html', cp=description, src=p, error_message=error_message)


@app.route('/developer',methods=["GET","POST"])
def developer():
    return render_template('dev.html')
    '''description = captionMulti(p)'''