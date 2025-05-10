from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from model import DropoutPredictor
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///student_dropout.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize the AI model
predictor = DropoutPredictor()
MODEL_PATH = 'dropout_model.joblib'

# Load the model if it exists
if os.path.exists(MODEL_PATH):
    try:
        predictor.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    students = db.relationship('Student', backref='faculty', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    gpa = db.Column(db.Float)
    attendance_rate = db.Column(db.Float)
    study_hours = db.Column(db.Float)
    previous_semester_grades = db.Column(db.String(20))
    family_income = db.Column(db.Float)
    dropout_risk = db.Column(db.Float)
    risk_factors = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    faculty_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'student_id': self.student_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'gpa': self.gpa,
            'attendance_rate': self.attendance_rate,
            'study_hours': self.study_hours,
            'previous_semester_grades': self.previous_semester_grades,
            'family_income': self.family_income,
            'dropout_risk': self.dropout_risk,
            'risk_factors': self.risk_factors.split(', ') if self.risk_factors else [],
            'notes': self.notes
        }

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Get all students but filter for at-risk in the template
    students = Student.query.filter_by(faculty_id=current_user.id).all()
    
    # Calculate statistics
    total_students = len(students)
    high_risk_count = len([s for s in students if s.dropout_risk and s.dropout_risk > 0.7])
    medium_risk_count = len([s for s in students if s.dropout_risk and 0.4 < s.dropout_risk <= 0.7])
    low_risk_count = len([s for s in students if s.dropout_risk and s.dropout_risk <= 0.4])
    
    # Calculate average metrics
    avg_gpa = sum(s.gpa for s in students) / total_students if total_students > 0 else 0
    avg_attendance = sum(s.attendance_rate for s in students) / total_students if total_students > 0 else 0
    avg_study_hours = sum(s.study_hours for s in students) / total_students if total_students > 0 else 0
    
    # Convert students to JSON-serializable format
    students_data = []
    for student in students:
        student_dict = student.to_dict()
        # Add additional fields needed for the dashboard
        student_dict['risk_score'] = student.dropout_risk
        students_data.append(student_dict)
    
    return render_template('dashboard.html',
                         students=students_data,
                         total_students=total_students,
                         high_risk_count=high_risk_count,
                         medium_risk_count=medium_risk_count,
                         low_risk_count=low_risk_count,
                         avg_gpa=avg_gpa,
                         avg_attendance=avg_attendance,
                         avg_study_hours=avg_study_hours)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        try:
            student = Student(
                student_id=request.form['student_id'],
                name=request.form['name'],
                age=int(request.form['age']),
                gender=request.form['gender'],
                gpa=float(request.form['gpa']),
                attendance_rate=float(request.form['attendance_rate']),
                study_hours=float(request.form['study_hours']),
                previous_semester_grades=request.form['previous_semester_grades'],
                family_income=float(request.form['family_income']),
                faculty_id=current_user.id
            )
            
            # Calculate dropout risk (placeholder for ML model)
            risk_factors = []
            risk_score = 0
            
            if student.gpa < 2.0:
                risk_factors.append("Low GPA")
                risk_score += 0.3
            if student.attendance_rate < 0.7:
                risk_factors.append("Poor Attendance")
                risk_score += 0.2
            if student.study_hours < 10:
                risk_factors.append("Low Study Hours")
                risk_score += 0.15
            if student.family_income < 30000:
                risk_factors.append("Low Family Income")
                risk_score += 0.15
            if 'F' in student.previous_semester_grades:
                risk_factors.append("Failed Courses")
                risk_score += 0.2
                
            student.dropout_risk = min(risk_score, 1.0)
            student.risk_factors = ", ".join(risk_factors)
            
            db.session.add(student)
            db.session.commit()
            flash('Student added successfully!')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'Error adding student: {str(e)}')
    return render_template('add_student.html')

@app.route('/upload_students', methods=['GET', 'POST'])
@login_required
def upload_students():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
                
                # Validate required columns
                required_columns = ['student_id', 'name', 'age', 'gender', 'gpa', 
                                 'attendance_rate', 'study_hours', 'previous_semester_grades', 
                                 'family_income']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    flash(f'Missing required columns: {", ".join(missing_columns)}')
                    return redirect(request.url)
                
                # Convert data types
                df['student_id'] = df['student_id'].astype(str)
                df['name'] = df['name'].astype(str)
                df['age'] = df['age'].astype(int)
                df['gender'] = df['gender'].astype(str)
                df['gpa'] = df['gpa'].astype(float)
                df['attendance_rate'] = df['attendance_rate'].astype(float)
                df['study_hours'] = df['study_hours'].astype(float)
                df['previous_semester_grades'] = df['previous_semester_grades'].astype(str)
                df['family_income'] = df['family_income'].astype(float)
                
                # Validate data ranges
                if not ((df['gpa'] >= 0) & (df['gpa'] <= 4.0)).all():
                    flash('GPA must be between 0 and 4.0')
                    return redirect(request.url)
                    
                if not ((df['attendance_rate'] >= 0) & (df['attendance_rate'] <= 1)).all():
                    flash('Attendance rate must be between 0 and 1')
                    return redirect(request.url)
                
                # Process each row
                for _, row in df.iterrows():
                    try:
                        student = Student(
                            student_id=str(row['student_id']),
                            name=str(row['name']),
                            age=int(row['age']),
                            gender=str(row['gender']),
                            gpa=float(row['gpa']),
                            attendance_rate=float(row['attendance_rate']),
                            study_hours=float(row['study_hours']),
                            previous_semester_grades=str(row['previous_semester_grades']),
                            family_income=float(row['family_income']),
                            faculty_id=current_user.id
                        )
                        
                        # Calculate dropout risk
                        risk_factors = []
                        risk_score = 0
                        
                        if student.gpa < 2.0:
                            risk_factors.append("Low GPA")
                            risk_score += 0.3
                        elif student.gpa < 2.5:
                            risk_factors.append("Below Average GPA")
                            risk_score += 0.15
                            
                        if student.attendance_rate < 0.7:
                            risk_factors.append("Poor Attendance")
                            risk_score += 0.2
                        elif student.attendance_rate < 0.8:
                            risk_factors.append("Below Average Attendance")
                            risk_score += 0.1
                            
                        if student.study_hours < 10:
                            risk_factors.append("Low Study Hours")
                            risk_score += 0.15
                        elif student.study_hours < 15:
                            risk_factors.append("Below Average Study Hours")
                            risk_score += 0.05
                            
                        if student.family_income < 30000:
                            risk_factors.append("Low Family Income")
                            risk_score += 0.15
                        elif student.family_income < 45000:
                            risk_factors.append("Below Average Family Income")
                            risk_score += 0.05
                            
                        if 'F' in student.previous_semester_grades:
                            risk_factors.append("Failed Courses")
                            risk_score += 0.2
                        elif 'D' in student.previous_semester_grades:
                            risk_factors.append("Poor Previous Grades")
                            risk_score += 0.1
                            
                        student.dropout_risk = min(risk_score, 1.0)
                        student.risk_factors = ", ".join(risk_factors)
                        
                        db.session.add(student)
                        
                    except Exception as e:
                        flash(f'Error processing row for student {row["student_id"]}: {str(e)}')
                        continue
                
                db.session.commit()
                flash('Students uploaded successfully!')
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)
                
    return render_template('upload_students.html')

@app.route('/student/<student_id>/report')
@login_required
def student_report(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            flash('Student not found. The student may have been removed or the ID is invalid.', 'error')
            return redirect(url_for('dashboard'))

        return render_template('student_report.html', student=student)

    except Exception as e:
        flash(f'Error viewing report: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/import_dummy_data')
@login_required
def import_dummy_data():
    try:
        # Check if dummy data file exists
        if not os.path.exists('dummy_students.csv'):
            flash('Dummy data file not found. Please run generate_dummy_data.py first.')
            return redirect(url_for('dashboard'))
            
        # Read dummy data
        df = pd.read_csv('dummy_students.csv')
        
        # Clear existing students for this faculty
        Student.query.filter_by(faculty_id=current_user.id).delete()
        db.session.commit()
        
        # Add students to database
        students_added = 0
        for _, row in df.iterrows():
            try:
                student = Student(
                    student_id=str(row['student_id']),
                    name=str(row['name']),
                    age=int(float(row['age'])),
                    gender=str(row['gender']),
                    gpa=float(row['gpa']),
                    attendance_rate=float(row['attendance_rate']),
                    study_hours=float(row['study_hours']),
                    previous_semester_grades=str(row['previous_semester_grades']),
                    family_income=float(row['family_income']),
                    dropout_risk=float(row['dropout_risk']),
                    risk_factors=str(row['risk_factors']),
                    faculty_id=current_user.id
                )
                db.session.add(student)
                students_added += 1
            except Exception as e:
                print(f"Error adding student {row['student_id']}: {str(e)}")
                continue
        
        db.session.commit()
        flash(f'Successfully imported {students_added} students!')
    except Exception as e:
        db.session.rollback()
        flash(f'Error importing dummy data: {str(e)}')
        print(f"Import error: {str(e)}")
    
    return redirect(url_for('dashboard'))

@app.route('/student/<student_id>/predict', methods=['POST'])
@login_required
def predict_student_risk(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found. The student may have been removed or the ID is invalid.'
            }), 404

        # Get latest student data
        student_data = {
            'gpa': student.gpa,
            'attendance_rate': student.attendance_rate,
            'study_hours': student.study_hours,
            'family_income': student.family_income,
            'previous_semester_grades': student.previous_semester_grades
        }

        # Calculate risk factors
        risk_factors = []
        risk_score = 0

        if student.gpa < 2.0:
            risk_factors.append("Low GPA")
            risk_score += 0.3
        if student.attendance_rate < 0.7:
            risk_factors.append("Poor Attendance")
            risk_score += 0.2
        if student.study_hours < 10:
            risk_factors.append("Low Study Hours")
            risk_score += 0.15
        if student.family_income < 30000:
            risk_factors.append("Low Family Income")
            risk_score += 0.15
        if 'F' in student.previous_semester_grades:
            risk_factors.append("Failed Courses")
            risk_score += 0.2

        # Update student record with new risk assessment
        student.dropout_risk = min(risk_score, 1.0)
        student.risk_factors = ", ".join(risk_factors)
        db.session.commit()

        return jsonify({
            'success': True,
            'risk_score': student.dropout_risk,
            'risk_factors': risk_factors
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error predicting risk: {str(e)}'
        }), 500

@app.route('/student/<student_id>/alert', methods=['POST'])
@login_required
def send_student_alert(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found'
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        alert_type = data.get('alert_type')
        message = data.get('message')

        if not alert_type or not message:
            return jsonify({
                'success': False,
                'message': 'Alert type and message are required'
            }), 400

        # Here you would typically send an actual alert (email, SMS, etc.)
        # For now, we'll just log it and store it in the notes
        alert_message = f"[{alert_type.upper()}] {message}"
        if student.notes:
            student.notes += f"\n{alert_message}"
        else:
            student.notes = alert_message

        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Alert sent successfully to {student.name}'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/student/<student_id>/history')
@login_required
def student_history(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            flash('Student not found. The student may have been removed or the ID is invalid.', 'error')
            return redirect(url_for('dashboard'))

        # Get student's history (you would typically have a separate History model)
        # For now, we'll just show the current data
        return render_template('student_history.html', student=student)

    except Exception as e:
        flash(f'Error viewing history: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/student/<student_id>/status', methods=['GET'])
@login_required
def get_student_status(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found. The student may have been removed or the ID is invalid.'
            }), 404

        return jsonify({
            'success': True,
            'status': student.status if hasattr(student, 'status') else 'active',
            'notes': student.notes if hasattr(student, 'notes') else ''
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting status: {str(e)}'
        }), 500

@app.route('/student/<student_id>/update_status', methods=['POST'])
@login_required
def update_student_status(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found. The student may have been removed or the ID is invalid.'
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        # Update student status and notes
        if hasattr(student, 'status'):
            student.status = data.get('status', student.status)
        if hasattr(student, 'notes'):
            student.notes = data.get('notes', student.notes)

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Status updated successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Error updating status: {str(e)}'
        }), 500

@app.route('/student/<student_id>/notifications', methods=['GET'])
@login_required
def get_student_notifications(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found. The student may have been removed or the ID is invalid.'
            }), 404

        # Get notifications (you would typically have a separate Notifications model)
        # For now, we'll return some dummy notifications
        notifications = [
            {
                'title': 'Risk Assessment Update',
                'message': f'Risk score updated to {student.dropout_risk * 100}%',
                'date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'alert' if student.dropout_risk > 0.7 else 'info'
            }
        ]

        return jsonify({
            'success': True,
            'notifications': notifications
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting notifications: {str(e)}'
        }), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()
        if not data or 'students' not in data:
            return jsonify({'error': 'No student data provided'}), 400

        students = data['students']
        if not students:
            return jsonify({'error': 'No students found to export'}), 400

        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Add title
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        elements.append(Paragraph("At-Risk Students Report", title_style))
        
        # Add timestamp
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            alignment=1
        )
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            timestamp_style
        ))
        elements.append(Spacer(1, 20))

        # Add summary
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20
        )
        elements.append(Paragraph(
            f"Total At-Risk Students: {len(students)}",
            summary_style
        ))
        elements.append(Spacer(1, 20))

        # Create table data
        table_data = [['Student ID', 'Name', 'GPA', 'Attendance', 'Risk Score', 'Risk Factors']]
        for student in students:
            table_data.append([
                student['student_id'],
                student['name'],
                f"{student['gpa']:.2f}",
                f"{student['attendance_rate']*100:.1f}%",
                f"{student['risk_score']*100:.1f}%",
                ', '.join(student['risk_factors'])
            ])

        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 30))

        # Add recommendations
        recommendations_style = ParagraphStyle(
            'Recommendations',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=10
        )
        elements.append(Paragraph("Recommendations", recommendations_style))
        
        recommendations = [
            "1. Schedule regular meetings with at-risk students",
            "2. Provide additional academic support and resources",
            "3. Monitor attendance and academic progress closely",
            "4. Consider implementing a mentoring program",
            "5. Review and adjust study plans as needed"
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(rec, styles['Normal']))
            elements.append(Spacer(1, 5))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='at_risk_students_report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        app.logger.error(f"Error generating PDF: {str(e)}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

@app.route('/student/<student_id>/view')
@login_required
def view_student(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found'
            }), 404

        return render_template('student_details.html', student=student)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/student/<student_id>/update', methods=['POST'])
@login_required
def update_student(student_id):
    try:
        student = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found'
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        # Update student fields
        if 'gpa' in data:
            student.gpa = float(data['gpa'])
        if 'attendance_rate' in data:
            student.attendance_rate = float(data['attendance_rate'])
        if 'study_hours' in data:
            student.study_hours = float(data['study_hours'])
        if 'notes' in data:
            student.notes = data['notes']

        # Recalculate risk score
        risk_factors = []
        risk_score = 0

        if student.gpa < 2.0:
            risk_factors.append("Low GPA")
            risk_score += 0.3
        elif student.gpa < 2.5:
            risk_factors.append("Below Average GPA")
            risk_score += 0.15

        if student.attendance_rate < 0.7:
            risk_factors.append("Poor Attendance")
            risk_score += 0.2
        elif student.attendance_rate < 0.8:
            risk_factors.append("Below Average Attendance")
            risk_score += 0.1

        if student.study_hours < 10:
            risk_factors.append("Low Study Hours")
            risk_score += 0.15
        elif student.study_hours < 15:
            risk_factors.append("Below Average Study Hours")
            risk_score += 0.05

        student.dropout_risk = min(risk_score, 1.0)
        student.risk_factors = ", ".join(risk_factors)

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Student updated successfully',
            'student': student.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=8080, host='localhost') 