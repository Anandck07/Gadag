import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

def generate_dummy_data(n_students=100):
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)

        # Generate student IDs
        student_ids = [f'STU{i:04d}' for i in range(1, n_students + 1)]

        # Generate names
        first_names = ['John', 'Jane', 'Michael', 'Emily', 'David', 'Sarah', 'James', 'Emma', 'William', 'Olivia',
                      'Daniel', 'Sophia', 'Matthew', 'Ava', 'Andrew', 'Isabella', 'Joseph', 'Mia', 'Christopher', 'Charlotte',
                      'Ethan', 'Amelia', 'Alexander', 'Harper', 'Benjamin', 'Evelyn', 'Elijah', 'Abigail', 'Lucas', 'Elizabeth',
                      'Mason', 'Sofia', 'Logan', 'Avery', 'Jacob', 'Ella', 'Jackson', 'Scarlett', 'Sebastian', 'Grace',
                      'Jack', 'Victoria', 'Owen', 'Riley', 'Gabriel', 'Aria', 'Matthew', 'Lily', 'Leo', 'Aubrey']
        
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                     'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                     'Lee', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
                     'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores', 'Green',
                     'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter', 'Roberts', 'Gomez']

        names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n_students)]

        # Generate other attributes
        ages = np.random.randint(18, 25, n_students)
        genders = np.random.choice(['Male', 'Female', 'Other'], n_students, p=[0.45, 0.45, 0.1])
        
        # Generate GPAs with a more realistic distribution
        gpas = np.random.normal(2.8, 0.8, n_students)
        gpas = np.clip(gpas, 0, 4.0)  # Clip GPA between 0 and 4.0
        
        # Generate attendance rates with more variation
        attendance_rates = np.random.normal(0.85, 0.15, n_students)
        attendance_rates = np.clip(attendance_rates, 0, 1)  # Clip attendance between 0 and 1
        
        # Generate study hours with more realistic distribution
        study_hours = np.random.normal(15, 7, n_students)
        study_hours = np.clip(study_hours, 0, 30)  # Clip study hours between 0 and 30

        # Generate previous semester grades with more realistic distribution
        grade_options = ['A', 'B', 'C', 'D', 'F']
        grade_weights = [0.2, 0.35, 0.25, 0.15, 0.05]  # More realistic grade distribution
        previous_grades = [random.choices(grade_options, weights=grade_weights, k=1)[0] for _ in range(n_students)]

        # Generate family income with more variation
        family_income = np.random.normal(60, 25, n_students)
        family_income = np.clip(family_income, 20, 200)  # Clip income between 20k and 200k

        # Create DataFrame
        df = pd.DataFrame({
            'student_id': student_ids,
            'name': names,
            'age': ages,
            'gender': genders,
            'gpa': gpas,
            'attendance_rate': attendance_rates,
            'study_hours': study_hours,
            'previous_semester_grades': previous_grades,
            'family_income': family_income
        })

        # Round numeric columns
        df['gpa'] = df['gpa'].round(2)
        df['attendance_rate'] = df['attendance_rate'].round(2)
        df['study_hours'] = df['study_hours'].round(1)
        df['family_income'] = df['family_income'].round(2)

        # Save to CSV
        df.to_csv('student_data.csv', index=False)
        print(f"Generated {n_students} student records and saved to student_data.csv")
        return True

    except Exception as e:
        print(f"Error generating data: {str(e)}")
        return False

if __name__ == "__main__":
    generate_dummy_data() 