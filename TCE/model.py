import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DropoutPredictor:
    def __init__(self):
        self.weights = {
            'gpa': 0.3,
            'attendance_rate': 0.2,
            'study_hours': 0.15,
            'previous_education': 0.1,
            'family_income': 0.1,
            'hours_worked': 0.15
        }
        self.thresholds = {
            'gpa': 2.0,
            'attendance_rate': 0.7,
            'study_hours': 10,
            'family_income': 30000,
            'hours_worked': 20
        }
        
    def normalize_feature(self, value, feature):
        """Normalize feature values to 0-1 range"""
        if feature == 'gpa':
            return max(0, min(1, value / 4.0))
        elif feature == 'attendance_rate':
            return value
        elif feature == 'study_hours':
            return max(0, min(1, value / 40))
        elif feature == 'family_income':
            return max(0, min(1, value / 100000))
        elif feature == 'hours_worked':
            return max(0, min(1, value / 40))
        elif feature == 'previous_education':
            education_levels = {
                'High School': 0.2,
                'Associate Degree': 0.4,
                'Bachelor Degree': 0.6,
                'Master Degree': 0.8
            }
            return education_levels.get(value, 0.2)
        return 0.5

    def predict(self, student_data):
        """Predict dropout risk for a single student"""
        try:
            risk_score = 0
            risk_factors = []
            
            # Calculate risk score based on weighted features
            for feature, weight in self.weights.items():
                value = student_data.get(feature, 0)
                normalized_value = self.normalize_feature(value, feature)
                
                # Add to risk score (higher normalized value = lower risk)
                risk_score += weight * (1 - normalized_value)
                
                # Check if feature is below threshold
                if feature in self.thresholds and value < self.thresholds[feature]:
                    risk_factors.append(f"Low {feature.replace('_', ' ').title()}")
            
            # Add additional risk factors
            if student_data.get('gpa', 0) < 2.0:
                risk_factors.append("Low GPA")
            if student_data.get('attendance_rate', 0) < 0.7:
                risk_factors.append("Poor Attendance")
            if student_data.get('study_hours', 0) < 10:
                risk_factors.append("Insufficient Study Hours")
            if student_data.get('hours_worked', 0) > 20:
                risk_factors.append("High Workload")
            
            return {
                'will_dropout': risk_score > 0.5,
                'risk_score': float(risk_score),
                'confidence': float(abs(risk_score - 0.5) * 2),
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def train(self, data):
        """Update weights based on historical data"""
        try:
            # Calculate average values for each feature
            averages = {}
            for feature in self.weights.keys():
                if feature in data.columns:
                    averages[feature] = data[feature].mean()
            
            # Adjust weights based on feature importance
            total_importance = 0
            for feature, avg in averages.items():
                if feature in self.weights:
                    # Higher standard deviation = more important feature
                    std = data[feature].std()
                    self.weights[feature] = std
                    total_importance += std
            
            # Normalize weights
            if total_importance > 0:
                for feature in self.weights:
                    self.weights[feature] /= total_importance
            
            logger.info("Model weights updated successfully")
            logger.info(f"New weights: {self.weights}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self, model_path='dropout_model.json'):
        """Save the model weights"""
        try:
            model_data = {
                'weights': self.weights,
                'thresholds': self.thresholds,
                'last_updated': datetime.now().isoformat()
            }
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=4)
            logger.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path='dropout_model.json'):
        """Load the model weights"""
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            self.weights = model_data['weights']
            self.thresholds = model_data['thresholds']
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_risk_factors(self, student_data):
        """Analyze and return risk factors for a student"""
        try:
            risk_factors = []
            
            # Check each feature against thresholds
            for feature, threshold in self.thresholds.items():
                value = student_data.get(feature, 0)
                if value < threshold:
                    risk_factors.append(f"Low {feature.replace('_', ' ').title()}")
            
            # Add additional risk factors
            if student_data.get('gpa', 0) < 2.0:
                risk_factors.append("Low GPA")
            if student_data.get('attendance_rate', 0) < 0.7:
                risk_factors.append("Poor Attendance")
            if student_data.get('study_hours', 0) < 10:
                risk_factors.append("Insufficient Study Hours")
            if student_data.get('hours_worked', 0) > 20:
                risk_factors.append("High Workload")
            
            return list(set(risk_factors))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {str(e)}")
            raise 