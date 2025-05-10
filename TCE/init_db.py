from app import app, db, Student, User
from datetime import datetime

def init_db():
    with app.app_context():
        # Drop all existing tables
        db.drop_all()
        
        # Create all tables
        db.create_all()
        
        # Create a test admin user
        admin = User(
            username='admin',
            email='admin@example.com',
            is_admin=True
        )
        admin.set_password('admin123')
        
        # Create a test student
        student = Student(
            student_id='TEST001',
            name='Test Student',
            age=20,
            gender='Male',
            gpa=3.5,
            attendance_rate=0.85,
            study_hours=15.0,
            previous_semester_grades='A,B,A',
            family_income=50000.0,
            dropout_risk=0.2,
            risk_factors='Good Performance',
            notes='Test student record',
            created_at=datetime.utcnow(),
            faculty_id=1
        )
        
        # Add and commit
        db.session.add(admin)
        db.session.add(student)
        db.session.commit()
        
        print("Database initialized successfully with test data")

if __name__ == '__main__':
    init_db() 