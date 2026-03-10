from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import os
import cv2
from werkzeug.utils import secure_filename

import pandas as pd
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
# ---------------------------------
# CONFIGURACIÓN
# ---------------------------------
import pyodbc

app = Flask(__name__)

def create_database_if_not_exists():

    connection = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=LAPTOP-HKAMDC53;"
        "Database=master;"
        "Trusted_Connection=yes;"
    )

    cursor = connection.cursor()

    cursor.execute("""
    IF DB_ID('myapp_db') IS NULL
        CREATE DATABASE myapp_db
    """)

    connection.commit()
    connection.close()


create_database_if_not_exists()

app.config['SECRET_KEY'] = 'mysecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = (
    'mssql+pyodbc:///?odbc_connect='
    'Driver={ODBC Driver 17 for SQL Server};'
    'Server=LAPTOP-HKAMDC53;'
    'Database=myapp_db;'
    'Trusted_Connection=yes;'
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ---------------------------------
# EXTENSIONES
# ---------------------------------

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ---------------------------------
# MODELOS YOLO
# ---------------------------------
from ultralytics import YOLO

# Modelo de detección de plaquetas
model = YOLO("model/best.pt")

# Modelo clasificador de imágenes
classifier = YOLO("model/frotis_classifier.pt")


# -------------------------------------------
# CLASIFICADOR DE IMAGEN (FROTIS - NO FROTIS)
# -------------------------------------------
def is_blood_smear(image_path):
    
    results = classifier(image_path)

    predicted_class = results[0].names[results[0].probs.top1]
    confidence = float(results[0].probs.top1conf)

    print(f"Clasificación: {predicted_class} | Confianza: {confidence}")

    if predicted_class.lower() == "frotis" and confidence > 0.75:
        return True

    return False

# ---------------------------------
# FUNCIONES MÉDICAS
# ---------------------------------

def estimate_platelets_per_ul(platelet_count):
    return platelet_count * 15000


def classify_dengue_risk(platelets_ul):
    if platelets_ul >= 150000:
        return "Normal"
    elif 100000 <= platelets_ul < 150000:
        return "Riesgo leve"
    elif 50000 <= platelets_ul < 100000:
        return "Posible dengue"
    else:
        return "Dengue severo"


# ---------------------------------
# MODELOS
# ---------------------------------

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(150), unique=True, nullable=False)

    email = db.Column(db.String(150), unique=True, nullable=False)

    password = db.Column(db.String(150), nullable=False)


class PlateletResult(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    image_filename = db.Column(db.String(150), nullable=False)

    platelet_count = db.Column(db.Integer, nullable=False)

    platelets_estimated = db.Column(db.Integer, nullable=True)

    dengue_status = db.Column(db.String(50), nullable=False)

    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('results', lazy=True))

# ---------------------------------
# CREAR TABLAS Y COLUMNAS
# ---------------------------------

with app.app_context():

    db.create_all()

with app.app_context():

    admin_user = User.query.filter_by(username="admin").first()

    if not admin_user:

        hashed_password = bcrypt.generate_password_hash("admin").decode('utf-8')

        admin = User(
            username="admin",
            email="admin@system.local",
            password=hashed_password
        )

        db.session.add(admin)
        db.session.commit()

        print("Usuario administrador creado: admin / admin")
# ---------------------------------
# LOGIN MANAGER
# ---------------------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------
# UTILIDADES
# ---------------------------------

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------
# RUTAS
# ---------------------------------

@app.route('/')
def index():
    return redirect(url_for('login'))


# ---------------------------------
# REGISTRO
# ---------------------------------

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():

    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():

            flash('El usuario o correo ya existe', 'danger')

            return redirect(url_for('register'))

        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Usuario registrado exitosamente', 'success')

        return redirect(url_for('register'))

    return render_template('register.html')


# ---------------------------------
# LOGIN
# ---------------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):

            login_user(user)

            flash('Inicio de sesión exitoso', 'success')

            return redirect(url_for('dashboard'))

        flash('Credenciales incorrectas', 'danger')

    return render_template('login.html')


# ---------------------------------
# LOGOUT
# ---------------------------------

@app.route('/logout')
@login_required
def logout():

    logout_user()

    flash('Sesión cerrada exitosamente', 'success')

    return redirect(url_for('login'))


# ---------------------------------
# DASHBOARD
# ---------------------------------

@app.route('/dashboard')
@login_required
def dashboard():

    user_results = PlateletResult.query.filter_by(user_id=current_user.id).all()

    return render_template('dashboard.html', results=user_results)


# ---------------------------------
# DETECCIÓN DE PLAQUETAS
# ---------------------------------

@app.route('/count-platelets', methods=['POST'])
@login_required
def count_platelets():

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if 'image' not in request.files:

        flash('No hay archivo en la solicitud', 'danger')

        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':

        flash('No seleccionaste un archivo', 'danger')

        return redirect(request.url)

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # -------------------------------------------
        # VALIDAR SI LA IMAGEN ES UN FROTIS
        # -------------------------------------------

        if not is_blood_smear(file_path):

            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "error": "La imagen no corresponde a un frotis sanguíneo"
            }), 400
        

        # -------------------------------------------
        # DETECCIÓN DE PLAQUETAS
        # -------------------------------------------

        image = cv2.imread(file_path)
        results = model(image)
        platelet_count = 0

        for box in results[0].boxes:
            if int(box.cls[0]) == 1:
                platelet_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(
                    image,
                    "Platelet",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

        # guardar imagen procesada
        processed_filename = "detected_" + filename
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)

        cv2.imwrite(processed_path, image)

        platelets_estimated = estimate_platelets_per_ul(platelet_count)

        dengue_status = classify_dengue_risk(platelets_estimated)

        new_result = PlateletResult(

            image_filename=filename,

            platelet_count=platelet_count,

            platelets_estimated=platelets_estimated,

            dengue_status=dengue_status,

            user_id=current_user.id
        )

        db.session.add(new_result)
        db.session.commit()

        return jsonify({
            'platelets_detected': platelet_count,
            'estimated_platelets_per_ul': platelets_estimated,
            'dengue_status': dengue_status,
            'processed_image': "/static/uploads/" + processed_filename
        })

    flash('Archivo no permitido', 'danger')

    return redirect(request.url)


# ---------------------------------
# EXPORTAR CSV
# ---------------------------------

@app.route('/export-csv')
@login_required
def export_csv():

    results = PlateletResult.query.filter_by(user_id=current_user.id)\
                .order_by(PlateletResult.id.desc()).all()

    data = []

    for i, result in enumerate(results, start=1):

        data.append({

            'N° Registro': i,

            'Imagen Analizada': result.image_filename,

            'Fecha de Análisis': result.analysis_date.strftime("%d/%m/%Y %H:%M"),

            'Conteo de Plaquetas Detectadas': result.platelet_count,

            'Estimación de Plaquetas (por µL)': result.platelets_estimated,

            'Diagnóstico de Dengue': result.dengue_status

        })

    df = pd.DataFrame(data)

    file_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"reporte_plaquetas_{current_user.username}.csv"
    )

    df.to_csv(file_path, index=False, encoding='utf-8-sig', sep=';')
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        f"reporte_plaquetas_{current_user.username}.csv",
        as_attachment=True
    )

# ---------------------------------
# RESULTADOS
# ---------------------------------

@app.route('/results')
@login_required
def results():

    user_results = PlateletResult.query.filter_by(user_id=current_user.id).all()

    return render_template('results.html', results=user_results)

# ---------------------------------
# EXPORTAR PDF
# ---------------------------------

@app.route('/export-pdf/<int:result_id>')
@login_required
def export_pdf(result_id):

    result = PlateletResult.query.get_or_404(result_id)

    pdf_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"reporte_plaquetas_{result.id}.pdf"
    )

    # Imagen original (sin bounding boxes)
    original_image_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        result.image_filename
    )

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Reporte de Análisis de Plaquetas", styles['Title']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"<b>Fecha:</b> {result.analysis_date.strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    if os.path.exists(original_image_path):

        elements.append(Paragraph("<b>Imagen Analizada:</b>", styles['Normal']))
        elements.append(Spacer(1,10))

        img_original = Image(original_image_path, width=4*inch, height=3*inch)

        elements.append(img_original)

    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"<b>Conteo de Plaquetas Detectadas:</b> {result.platelet_count}", styles['Normal']))
    elements.append(Paragraph(f"<b>Estimación de Plaquetas (µL):</b> {result.platelets_estimated}", styles['Normal']))
    elements.append(Paragraph(f"<b>Diagnóstico:</b> {result.dengue_status}", styles['Normal']))
    elements.append(Spacer(1,30))

    image_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        "detected_" + result.image_filename
    )

    if os.path.exists(image_path):
        img = Image(image_path, width=4*inch, height=3*inch)
        elements.append(img)

    doc.build(elements)

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        f"reporte_plaquetas_{result.id}.pdf",
        as_attachment=True
    )

# ---------------------------------
# EJECUTAR
# ---------------------------------

if __name__ == '__main__':
    app.run(debug=True)