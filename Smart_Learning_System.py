"""
AI-Powered Personalized Learning System - Enhanced with Custom Dataset Support
Designing Smart Educational Experiences for Every Learner

Now supports loading custom learner data and quiz questions from external files!

Author: Kuldeep
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import random
from collections import defaultdict
import io

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Smart Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #1a365d;
    }
    
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    
    .profile-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 10px;
        color: #1a365d;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 15px;
        border-radius: 10px;
        color: #744210;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .challenge-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 4px 10px rgba(245, 87, 108, 0.3);
    }
    
    .stButton>button {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        font-family: 'Source Sans 3', sans-serif;
    }
    
    .stButton>button:hover {
        border-color: #667eea;
        background-color: #f7fafc;
        transform: scale(1.02);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .feedback-message {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        font-family: 'Source Sans 3', sans-serif;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CUSTOM DATASET LOADING FUNCTIONS ====================

def load_custom_learners_csv(uploaded_file):
    """
    Load learner data from CSV file
    Expected columns: learner_name, learner_id, engagement_score, learning_pace, classification
    """
    try:
        df = pd.read_csv(uploaded_file)
        learners_data = {}
        
        for _, row in df.iterrows():
            learner_name = row.get('learner_name', f"Learner_{row.get('learner_id', 'Unknown')}")
            learners_data[learner_name] = {
                "id": row.get('learner_id', f"L{random.randint(100, 999)}"),
                "quiz_history": [],
                "engagement_score": int(row.get('engagement_score', 70)),
                "learning_pace": row.get('learning_pace', 'moderate'),
                "classification": row.get('classification', 'Average Learner'),
                "strengths": row.get('strengths', 'Consistent Performance').split(',') if pd.notna(row.get('strengths')) else [],
                "weaknesses": row.get('weaknesses', 'Complex Problems').split(',') if pd.notna(row.get('weaknesses')) else []
            }
        
        st.success(f"✅ Loaded {len(learners_data)} learners from CSV!")
        return learners_data
    
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def load_custom_learners_json(uploaded_file):
    """
    Load learner data from JSON file
    Expected format: {"learner_name": {learner_data}, ...}
    """
    try:
        data = json.load(uploaded_file)
        st.success(f"✅ Loaded {len(data)} learners from JSON!")
        return data
    
    except Exception as e:
        st.error(f"Error loading JSON: {str(e)}")
        return None

def load_quiz_history_csv(uploaded_file):
    """
    Load quiz history from CSV file
    Expected columns: learner_name, accuracy, avg_time, hints_used, retries, date, topic
    """
    try:
        df = pd.read_csv(uploaded_file)
        quiz_history = defaultdict(list)
        
        for _, row in df.iterrows():
            learner_name = row.get('learner_name', 'Unknown')
            quiz_history[learner_name].append({
                "accuracy": float(row.get('accuracy', 70)),
                "avg_time": float(row.get('avg_time', 50)),
                "hints_used": int(row.get('hints_used', 0)),
                "retries": int(row.get('retries', 0)),
                "date": row.get('date', datetime.now().strftime("%Y-%m-%d")),
                "topic": row.get('topic', 'General')
            })
        
        st.success(f"✅ Loaded quiz history for {len(quiz_history)} learners!")
        return dict(quiz_history)
    
    except Exception as e:
        st.error(f"Error loading quiz history: {str(e)}")
        return None

def load_custom_questions_json(uploaded_file):
    """
    Load custom quiz questions from JSON
    Expected format: {"topic": {"difficulty": [questions]}}
    """
    try:
        questions = json.load(uploaded_file)
        st.success(f"✅ Loaded custom quiz questions!")
        return questions
    
    except Exception as e:
        st.error(f"Error loading questions: {str(e)}")
        return None

def load_custom_questions_csv(uploaded_file):
    """
    Load quiz questions from CSV
    Expected columns: topic, difficulty, question, option1, option2, option3, option4, correct_index, hint, explanation
    """
    try:
        df = pd.read_csv(uploaded_file)
        questions_db = defaultdict(lambda: defaultdict(list))
        
        for _, row in df.iterrows():
            topic = row.get('topic', 'General')
            difficulty = row.get('difficulty', 'medium')
            
            question_data = {
                "question": row.get('question', ''),
                "options": [
                    row.get('option1', ''),
                    row.get('option2', ''),
                    row.get('option3', ''),
                    row.get('option4', '')
                ],
                "correct": int(row.get('correct_index', 0)),
                "hint": row.get('hint', 'Think carefully about this question'),
                "explanation": row.get('explanation', 'Review the concept')
            }
            
            questions_db[topic][difficulty].append(question_data)
        
        st.success(f"✅ Loaded {sum(len(q) for t in questions_db.values() for q in t.values())} questions!")
        return dict(questions_db)
    
    except Exception as e:
        st.error(f"Error loading questions CSV: {str(e)}")
        return None

def export_learner_data_csv(learners_data):
    """Export current learner data to CSV format"""
    data = []
    for name, learner in learners_data.items():
        data.append({
            'learner_name': name,
            'learner_id': learner['id'],
            'engagement_score': learner['engagement_score'],
            'learning_pace': learner['learning_pace'],
            'classification': learner['classification'],
            'strengths': ','.join(learner['strengths']),
            'weaknesses': ','.join(learner['weaknesses'])
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def export_quiz_history_csv(learners_data):
    """Export quiz history to CSV format"""
    data = []
    for name, learner in learners_data.items():
        for quiz in learner['quiz_history']:
            data.append({
                'learner_name': name,
                'accuracy': quiz['accuracy'],
                'avg_time': quiz['avg_time'],
                'hints_used': quiz['hints_used'],
                'retries': quiz['retries'],
                'date': quiz['date'],
                'topic': quiz['topic']
            })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# ==================== DATA INITIALIZATION ====================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_learner = None
        st.session_state.learners_data = generate_sample_learners()
        st.session_state.custom_questions_db = None
        st.session_state.quiz_history = []
        st.session_state.current_quiz = None
        st.session_state.current_question_index = 0
        st.session_state.quiz_started = False
        st.session_state.feedback_messages = []
        st.session_state.content_recommendations = []
        st.session_state.learner_profile_updated = False
        st.session_state.using_custom_data = False

def generate_sample_learners():
    """Generate sample learner profiles with realistic data"""
    learners = {
        "Alice Johnson": {
            "id": "L001",
            "quiz_history": [
                {"accuracy": 45, "avg_time": 85, "hints_used": 8, "retries": 3, "date": "2026-01-10", "topic": "Algebra"},
                {"accuracy": 52, "avg_time": 78, "hints_used": 6, "retries": 2, "date": "2026-01-12", "topic": "Geometry"},
                {"accuracy": 48, "avg_time": 82, "hints_used": 7, "retries": 3, "date": "2026-01-14", "topic": "Algebra"}
            ],
            "engagement_score": 62,
            "learning_pace": "slow",
            "classification": "Struggling Learner",
            "strengths": ["Visual Learning", "Pattern Recognition"],
            "weaknesses": ["Abstract Reasoning", "Time Management"]
        },
        "Bob Smith": {
            "id": "L002",
            "quiz_history": [
                {"accuracy": 75, "avg_time": 45, "hints_used": 2, "retries": 0, "date": "2026-01-10", "topic": "Algebra"},
                {"accuracy": 78, "avg_time": 42, "hints_used": 1, "retries": 0, "date": "2026-01-12", "topic": "Geometry"},
                {"accuracy": 73, "avg_time": 48, "hints_used": 2, "retries": 1, "date": "2026-01-14", "topic": "Statistics"}
            ],
            "engagement_score": 78,
            "learning_pace": "moderate",
            "classification": "Average Learner",
            "strengths": ["Consistent Performance", "Good Understanding"],
            "weaknesses": ["Complex Problem Solving", "Advanced Concepts"]
        },
        "Carol Williams": {
            "id": "L003",
            "quiz_history": [
                {"accuracy": 92, "avg_time": 28, "hints_used": 0, "retries": 0, "date": "2026-01-10", "topic": "Algebra"},
                {"accuracy": 95, "avg_time": 25, "hints_used": 0, "retries": 0, "date": "2026-01-12", "topic": "Geometry"},
                {"accuracy": 94, "avg_time": 26, "hints_used": 0, "retries": 0, "date": "2026-01-14", "topic": "Calculus"}
            ],
            "engagement_score": 95,
            "learning_pace": "fast",
            "classification": "Advanced Learner",
            "strengths": ["Quick Comprehension", "Problem Solving", "Self-Directed"],
            "weaknesses": ["May skip fundamentals", "Needs challenges"]
        },
        "David Brown": {
            "id": "L004",
            "quiz_history": [
                {"accuracy": 68, "avg_time": 55, "hints_used": 3, "retries": 1, "date": "2026-01-10", "topic": "Algebra"},
                {"accuracy": 71, "avg_time": 52, "hints_used": 2, "retries": 1, "date": "2026-01-12", "topic": "Geometry"},
                {"accuracy": 69, "avg_time": 58, "hints_used": 3, "retries": 2, "date": "2026-01-14", "topic": "Algebra"}
            ],
            "engagement_score": 72,
            "learning_pace": "moderate",
            "classification": "Average Learner",
            "strengths": ["Persistence", "Improvement Mindset"],
            "weaknesses": ["Confidence", "Speed"]
        }
    }
    return learners

# ==================== QUIZ QUESTIONS DATABASE ====================

def get_quiz_questions(topic, difficulty):
    """Return quiz questions based on topic and difficulty level"""
    # Use custom questions if available
    if st.session_state.custom_questions_db:
        custom_questions = st.session_state.custom_questions_db.get(topic, {}).get(difficulty, [])
        if custom_questions:
            return custom_questions
    
    # Default questions database
    questions_db = {
        "Algebra": {
            "easy": [
                {
                    "question": "Solve for x: 2x + 5 = 13",
                    "options": ["x = 3", "x = 4", "x = 5", "x = 6"],
                    "correct": 1,
                    "hint": "Subtract 5 from both sides first, then divide by 2",
                    "explanation": "First, subtract 5 from both sides: 2x = 8. Then divide both sides by 2: x = 4"
                },
                {
                    "question": "What is 3(x + 2) expanded?",
                    "options": ["3x + 2", "3x + 5", "3x + 6", "x + 6"],
                    "correct": 2,
                    "hint": "Multiply 3 by each term inside the parentheses",
                    "explanation": "Using the distributive property: 3 × x = 3x and 3 × 2 = 6, so 3x + 6"
                },
                {
                    "question": "If y = 2x and x = 3, what is y?",
                    "options": ["5", "6", "7", "8"],
                    "correct": 1,
                    "hint": "Substitute x = 3 into the equation y = 2x",
                    "explanation": "y = 2 × 3 = 6"
                }
            ],
            "medium": [
                {
                    "question": "Solve: 3x - 7 = 2x + 5",
                    "options": ["x = 10", "x = 11", "x = 12", "x = 13"],
                    "correct": 2,
                    "hint": "Get all x terms on one side and constants on the other",
                    "explanation": "Subtract 2x from both sides: x - 7 = 5. Add 7 to both sides: x = 12"
                },
                {
                    "question": "Factor: x² + 5x + 6",
                    "options": ["(x+2)(x+3)", "(x+1)(x+6)", "(x+4)(x+2)", "(x+5)(x+1)"],
                    "correct": 0,
                    "hint": "Find two numbers that multiply to 6 and add to 5",
                    "explanation": "2 and 3 multiply to 6 and add to 5, so (x+2)(x+3)"
                },
                {
                    "question": "Simplify: (2x²y)(3xy²)",
                    "options": ["5x³y³", "6x³y³", "6x²y²", "5xy"],
                    "correct": 1,
                    "hint": "Multiply coefficients and add exponents of like bases",
                    "explanation": "2 × 3 = 6, x² × x = x³, y × y² = y³, giving 6x³y³"
                }
            ],
            "hard": [
                {
                    "question": "Solve the quadratic equation: x² - 7x + 12 = 0",
                    "options": ["x = 2, 5", "x = 3, 4", "x = 1, 6", "x = 2, 6"],
                    "correct": 1,
                    "hint": "Factor or use the quadratic formula",
                    "explanation": "(x-3)(x-4) = 0, so x = 3 or x = 4"
                },
                {
                    "question": "If f(x) = 2x² - 3x + 1, what is f(3)?",
                    "options": ["8", "10", "12", "14"],
                    "correct": 1,
                    "hint": "Substitute x = 3 into the function",
                    "explanation": "f(3) = 2(3)² - 3(3) + 1 = 2(9) - 9 + 1 = 18 - 9 + 1 = 10"
                },
                {
                    "question": "Solve for x: (x+2)² = 25",
                    "options": ["x = 3 or -7", "x = 5 or -5", "x = 2 or -2", "x = 3 or 7"],
                    "correct": 0,
                    "hint": "Take the square root of both sides, remembering ±",
                    "explanation": "x+2 = ±5, so x = 3 or x = -7"
                }
            ]
        },
        "Geometry": {
            "easy": [
                {
                    "question": "What is the area of a rectangle with length 8 and width 5?",
                    "options": ["13", "26", "40", "45"],
                    "correct": 2,
                    "hint": "Area = length × width",
                    "explanation": "Area = 8 × 5 = 40 square units"
                },
                {
                    "question": "How many degrees are in a right angle?",
                    "options": ["45°", "60°", "90°", "180°"],
                    "correct": 2,
                    "hint": "A right angle forms a perfect corner, like the letter L",
                    "explanation": "A right angle is exactly 90 degrees"
                },
                {
                    "question": "What is the perimeter of a square with side length 6?",
                    "options": ["12", "18", "24", "36"],
                    "correct": 2,
                    "hint": "Perimeter = 4 × side for a square",
                    "explanation": "Perimeter = 4 × 6 = 24 units"
                }
            ],
            "medium": [
                {
                    "question": "Find the area of a triangle with base 10 and height 6",
                    "options": ["16", "30", "60", "120"],
                    "correct": 1,
                    "hint": "Area = (1/2) × base × height",
                    "explanation": "Area = (1/2) × 10 × 6 = 30 square units"
                },
                {
                    "question": "What is the circumference of a circle with radius 7? (Use π ≈ 3.14)",
                    "options": ["21.98", "43.96", "153.86", "307.72"],
                    "correct": 1,
                    "hint": "Circumference = 2πr",
                    "explanation": "C = 2 × 3.14 × 7 = 43.96 units"
                },
                {
                    "question": "If two angles in a triangle are 45° and 60°, what is the third angle?",
                    "options": ["65°", "70°", "75°", "80°"],
                    "correct": 2,
                    "hint": "Angles in a triangle sum to 180°",
                    "explanation": "180° - 45° - 60° = 75°"
                }
            ],
            "hard": [
                {
                    "question": "Find the area of a circle with diameter 14 (Use π ≈ 3.14)",
                    "options": ["43.96", "153.86", "307.72", "615.44"],
                    "correct": 1,
                    "hint": "First find the radius (diameter ÷ 2), then use A = πr²",
                    "explanation": "r = 7, A = 3.14 × 7² = 3.14 × 49 = 153.86 square units"
                },
                {
                    "question": "A rectangular prism has dimensions 4×5×6. What is its volume?",
                    "options": ["60", "80", "100", "120"],
                    "correct": 3,
                    "hint": "Volume = length × width × height",
                    "explanation": "V = 4 × 5 × 6 = 120 cubic units"
                },
                {
                    "question": "What is the length of the hypotenuse in a right triangle with legs 6 and 8?",
                    "options": ["8", "10", "12", "14"],
                    "correct": 1,
                    "hint": "Use the Pythagorean theorem: a² + b² = c²",
                    "explanation": "6² + 8² = 36 + 64 = 100 = 10²; hypotenuse = 10"
                }
            ]
        },
        "Statistics": {
            "easy": [
                {
                    "question": "What is the mean of 5, 10, 15, 20?",
                    "options": ["10", "12.5", "15", "17.5"],
                    "correct": 1,
                    "hint": "Mean = sum of values ÷ number of values",
                    "explanation": "Mean = (5+10+15+20) ÷ 4 = 50 ÷ 4 = 12.5"
                },
                {
                    "question": "What is the median of 3, 7, 5, 9, 11?",
                    "options": ["5", "7", "9", "11"],
                    "correct": 1,
                    "hint": "Sort the numbers and find the middle value",
                    "explanation": "Sorted: 3, 5, 7, 9, 11. Middle value is 7"
                },
                {
                    "question": "What is the mode of 2, 3, 3, 5, 6, 3, 7?",
                    "options": ["2", "3", "5", "7"],
                    "correct": 1,
                    "hint": "Mode is the most frequently occurring value",
                    "explanation": "3 appears three times, more than any other value"
                }
            ],
            "medium": [
                {
                    "question": "Calculate the range of: 12, 18, 15, 22, 9, 25",
                    "options": ["13", "14", "15", "16"],
                    "correct": 3,
                    "hint": "Range = maximum value - minimum value",
                    "explanation": "Range = 25 - 9 = 16"
                },
                {
                    "question": "If the mean of 4 numbers is 15 and three of them are 10, 12, 18, what's the fourth?",
                    "options": ["18", "20", "22", "24"],
                    "correct": 1,
                    "hint": "Total sum = mean × count, then solve for the missing number",
                    "explanation": "Total = 15 × 4 = 60. Fourth number = 60 - 10 - 12 - 18 = 20"
                }
            ],
            "hard": [
                {
                    "question": "Calculate the variance of: 4, 8, 6, 5, 3, 9 (mean = 5.83)",
                    "options": ["4.47", "5.14", "6.81", "7.25"],
                    "correct": 0,
                    "hint": "Variance = average of squared differences from mean",
                    "explanation": "Variance ≈ 4.47 (detailed calculation involves squaring deviations)"
                }
            ]
        },
        "Calculus": {
            "medium": [
                {
                    "question": "What is the derivative of f(x) = 3x²?",
                    "options": ["3x", "6x", "x²", "3x³"],
                    "correct": 1,
                    "hint": "Use the power rule: d/dx(xⁿ) = n·xⁿ⁻¹",
                    "explanation": "Using power rule: 3 × 2 × x¹ = 6x"
                },
                {
                    "question": "What is ∫ 2x dx?",
                    "options": ["2x", "x²", "x² + C", "2x² + C"],
                    "correct": 2,
                    "hint": "Reverse the power rule and add constant C",
                    "explanation": "∫ 2x dx = x² + C"
                }
            ],
            "hard": [
                {
                    "question": "Find the derivative of f(x) = x³ - 4x² + 7x - 2",
                    "options": ["3x² - 8x + 7", "3x² - 4x + 7", "x² - 8x + 7", "3x³ - 8x + 7"],
                    "correct": 0,
                    "hint": "Apply power rule to each term separately",
                    "explanation": "f'(x) = 3x² - 8x + 7"
                },
                {
                    "question": "What is the limit of (x² - 4)/(x - 2) as x approaches 2?",
                    "options": ["0", "2", "4", "undefined"],
                    "correct": 2,
                    "hint": "Factor the numerator and simplify before substituting",
                    "explanation": "(x² - 4)/(x - 2) = (x+2)(x-2)/(x-2) = x + 2, so limit = 4"
                }
            ]
        }
    }
    
    return questions_db.get(topic, {}).get(difficulty, [])

# ==================== ML MODELS AND CLASSIFICATION ====================

def classify_learner(learner_data):
    """Classify learner based on performance metrics using ML"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return "New Learner"
    
    avg_accuracy = np.mean([q['accuracy'] for q in quiz_history])
    avg_time = np.mean([q['avg_time'] for q in quiz_history])
    total_hints = sum([q['hints_used'] for q in quiz_history])
    total_retries = sum([q['retries'] for q in quiz_history])
    
    if avg_accuracy < 60 and (avg_time > 70 or total_hints > 5):
        return "Struggling Learner"
    elif avg_accuracy >= 85 and avg_time < 35 and total_hints == 0:
        return "Advanced Learner"
    else:
        return "Average Learner"

def cluster_learners(learners_data):
    """Cluster learners using K-Means based on accuracy and pace"""
    if len(learners_data) < 3:
        return {}
    
    features = []
    learner_names = []
    
    for name, data in learners_data.items():
        if data['quiz_history']:
            avg_accuracy = np.mean([q['accuracy'] for q in data['quiz_history']])
            avg_time = np.mean([q['avg_time'] for q in data['quiz_history']])
            features.append([avg_accuracy, avg_time])
            learner_names.append(name)
    
    if len(features) >= 3:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        cluster_mapping = {}
        for name, cluster in zip(learner_names, clusters):
            cluster_mapping[name] = int(cluster)
        
        return cluster_mapping
    
    return {}

def predict_next_difficulty(learner_data):
    """Predict optimal difficulty level for next quiz"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return "easy"
    
    recent_performance = quiz_history[-3:] if len(quiz_history) >= 3 else quiz_history
    avg_accuracy = np.mean([q['accuracy'] for q in recent_performance])
    avg_time = np.mean([q['avg_time'] for q in recent_performance])
    
    if avg_accuracy >= 90 and avg_time < 35:
        return "hard"
    elif avg_accuracy >= 70 and avg_time < 60:
        return "medium"
    else:
        return "easy"

# ==================== CONTENT ADAPTATION ENGINE ====================

def adapt_content(learner_data):
    """Dynamically adapt content based on learner performance"""
    classification = learner_data['classification']
    quiz_history = learner_data['quiz_history']
    
    adaptations = {
        "difficulty": predict_next_difficulty(learner_data),
        "content_format": "visual" if classification == "Struggling Learner" else "text",
        "enable_challenge_mode": classification == "Advanced Learner",
        "provide_hints": classification != "Advanced Learner",
        "revision_needed": False
    }
    
    if quiz_history:
        recent_accuracy = quiz_history[-1]['accuracy'] if quiz_history else 0
        if recent_accuracy < 50:
            adaptations['revision_needed'] = True
            adaptations['difficulty'] = 'easy'
    
    return adaptations

def generate_personalized_feedback(question_data, user_answer, correct_answer, learner_classification):
    """Generate adaptive feedback based on learner level"""
    is_correct = user_answer == correct_answer
    
    if is_correct:
        if learner_classification == "Advanced Learner":
            messages = [
                "Excellent! Ready for more challenges?",
                "Perfect! Your problem-solving skills are sharp.",
                "Outstanding! Keep pushing your limits."
            ]
        elif learner_classification == "Average Learner":
            messages = [
                "Great job! You're making solid progress.",
                "Well done! Your understanding is improving.",
                "Correct! Keep up the consistent work."
            ]
        else:
            messages = [
                "Fantastic! See, you can do it! 🌟",
                "Wonderful! This is real progress!",
                "Yes! You're getting stronger at this!"
            ]
        return random.choice(messages), "success"
    else:
        hint = question_data.get('hint', 'Try to break down the problem step by step.')
        if learner_classification == "Struggling Learner":
            feedback = f"Not quite right, but don't give up! 💪 Here's a hint: {hint}"
        else:
            feedback = f"Not correct. Think about: {hint}"
        return feedback, "error"

def generate_recommendations(learner_data, current_topic):
    """Generate personalized learning recommendations"""
    classification = learner_data['classification']
    
    recommendations = []
    
    if classification == "Struggling Learner":
        recommendations = [
            {
                "type": "📹 Video Tutorial",
                "title": f"Visual Guide to {current_topic} Basics",
                "description": "Step-by-step visual explanation with examples",
                "priority": "High"
            },
            {
                "type": "✏️ Guided Practice",
                "title": "Interactive Practice Problems",
                "description": "Practice with instant feedback and hints",
                "priority": "High"
            },
            {
                "type": "📝 Revision Summary",
                "title": f"{current_topic} Key Concepts Review",
                "description": "Quick reference sheet with formulas and examples",
                "priority": "Medium"
            },
            {
                "type": "🎯 Focus Session",
                "title": "One-on-One Tutoring Recommended",
                "description": "Personalized help with difficult concepts",
                "priority": "High"
            }
        ]
    elif classification == "Advanced Learner":
        recommendations = [
            {
                "type": "🏆 Challenge Problem",
                "title": f"Advanced {current_topic} Competition Problems",
                "description": "Olympiad-level questions to test your skills",
                "priority": "High"
            },
            {
                "type": "🔬 Research Project",
                "title": "Real-World Application Project",
                "description": "Apply concepts to solve real problems",
                "priority": "Medium"
            },
            {
                "type": "👥 Peer Teaching",
                "title": "Help Others Learn",
                "description": "Reinforce your knowledge by teaching peers",
                "priority": "Medium"
            },
            {
                "type": "📚 Advanced Topics",
                "title": f"Next Level: Beyond {current_topic}",
                "description": "Explore university-level concepts",
                "priority": "High"
            }
        ]
    else:
        recommendations = [
            {
                "type": "📝 Practice Set",
                "title": f"{current_topic} Mixed Practice",
                "description": "Variety of problems to strengthen skills",
                "priority": "High"
            },
            {
                "type": "📹 Concept Review",
                "title": "Video Review of Key Topics",
                "description": "Refresh your understanding",
                "priority": "Medium"
            },
            {
                "type": "🎯 Skill Builder",
                "title": "Targeted Improvement Exercises",
                "description": "Focus on areas needing work",
                "priority": "High"
            },
            {
                "type": "⚡ Quick Quiz",
                "title": "Daily Challenge",
                "description": "Keep your skills sharp with daily practice",
                "priority": "Medium"
            }
        ]
    
    return recommendations

# ==================== VISUALIZATION FUNCTIONS ====================

def create_performance_chart(learner_data):
    """Create interactive performance over time chart"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return None
    
    df = pd.DataFrame(quiz_history)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['accuracy'],
        mode='lines+markers',
        name='Accuracy (%)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#667eea')
    ))
    
    fig.update_layout(
        title="Performance Trend",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=300
    )
    
    return fig

def create_topic_heatmap(learner_data):
    """Create heatmap of topic-wise performance"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return None
    
    topic_performance = defaultdict(list)
    for quiz in quiz_history:
        topic_performance[quiz['topic']].append(quiz['accuracy'])
    
    topics = list(topic_performance.keys())
    avg_scores = [np.mean(scores) for scores in topic_performance.values()]
    
    fig = go.Figure(data=go.Bar(
        x=topics,
        y=avg_scores,
        marker=dict(
            color=avg_scores,
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            showscale=True
        ),
        text=[f"{score:.1f}%" for score in avg_scores],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Topic-Wise Strengths",
        xaxis_title="Topic",
        yaxis_title="Average Accuracy (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=300
    )
    
    return fig

def create_metrics_gauge(value, title, max_value=100):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_value/3], 'color': "#ffecd2"},
                {'range': [max_value/3, 2*max_value/3], 'color': "#fdcb6e"},
                {'range': [2*max_value/3, max_value], 'color': "#84fab0"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif")
    )
    
    return fig

def create_time_vs_accuracy_scatter(learner_data):
    """Create scatter plot of time vs accuracy"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return None
    
    df = pd.DataFrame(quiz_history)
    
    fig = px.scatter(
        df, 
        x='avg_time', 
        y='accuracy',
        color='topic',
        size=[15] * len(df),
        hover_data=['date'],
        title='Time vs Accuracy Analysis'
    )
    
    fig.update_layout(
        xaxis_title="Average Time (seconds)",
        yaxis_title="Accuracy (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=350
    )
    
    return fig

def create_hints_usage_chart(learner_data):
    """Create bar chart for hints usage over time"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return None
    
    df = pd.DataFrame(quiz_history)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['date'],
            y=df['hints_used'],
            marker=dict(
                color=df['hints_used'],
                colorscale='Reds',
                showscale=True
            ),
            text=df['hints_used'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Hints Usage Over Time",
        xaxis_title="Date",
        yaxis_title="Hints Used",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=300
    )
    
    return fig

def create_improvement_trajectory(learner_data):
    """Create line chart showing improvement trajectory"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history or len(quiz_history) < 2:
        return None
    
    df = pd.DataFrame(quiz_history)
    
    # Calculate moving average
    df['ma_accuracy'] = df['accuracy'].rolling(window=min(3, len(df)), min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['accuracy'],
        mode='markers',
        name='Actual Score',
        marker=dict(size=8, color='#667eea', opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['ma_accuracy'],
        mode='lines',
        name='Trend',
        line=dict(color='#f093fb', width=3)
    ))
    
    fig.update_layout(
        title="Learning Progress Trajectory",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=350
    )
    
    return fig

def create_performance_distribution(learner_data):
    """Create histogram of performance distribution"""
    quiz_history = learner_data['quiz_history']
    
    if not quiz_history:
        return None
    
    accuracies = [q['accuracy'] for q in quiz_history]
    
    fig = go.Figure(data=[go.Histogram(
        x=accuracies,
        nbinsx=10,
        marker=dict(
            color='#667eea',
            line=dict(color='white', width=1)
        )
    )])
    
    fig.update_layout(
        title="Score Distribution",
        xaxis_title="Accuracy (%)",
        yaxis_title="Frequency",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Source Sans 3, sans-serif"),
        height=300
    )
    
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    """Main application logic"""
    
    initialize_session_state()
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #1a365d; font-size: 2.8em;'>🎓 Smart Learning System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #4a5568; font-weight: 300; margin-bottom: 30px;'>Designing Educational Experiences for Every Learner</h3>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Management")
        
        # Custom Dataset Upload Section
        with st.expander("📤 Upload Custom Data", expanded=False):
            st.markdown("**Upload Learner Profiles:**")
            learner_file = st.file_uploader(
                "CSV or JSON file",
                type=['csv', 'json'],
                key='learner_upload',
                help="CSV: learner_name, learner_id, engagement_score, learning_pace, classification"
            )
            
            if learner_file:
                if learner_file.name.endswith('.csv'):
                    custom_learners = load_custom_learners_csv(learner_file)
                else:
                    custom_learners = load_custom_learners_json(learner_file)
                
                if custom_learners:
                    if st.button("✅ Use This Data"):
                        st.session_state.learners_data = custom_learners
                        st.session_state.using_custom_data = True
                        st.rerun()
            
            st.markdown("---")
            st.markdown("**Upload Quiz History:**")
            quiz_file = st.file_uploader(
                "CSV file with quiz history",
                type=['csv'],
                key='quiz_upload',
                help="CSV: learner_name, accuracy, avg_time, hints_used, retries, date, topic"
            )
            
            if quiz_file:
                quiz_history = load_quiz_history_csv(quiz_file)
                if quiz_history:
                    if st.button("✅ Import Quiz History"):
                        for learner_name, history in quiz_history.items():
                            if learner_name in st.session_state.learners_data:
                                st.session_state.learners_data[learner_name]['quiz_history'].extend(history)
                                # Update classification
                                st.session_state.learners_data[learner_name]['classification'] = classify_learner(
                                    st.session_state.learners_data[learner_name]
                                )
                        st.rerun()
            
            st.markdown("---")
            st.markdown("**Upload Custom Questions:**")
            questions_file = st.file_uploader(
                "JSON or CSV file",
                type=['json', 'csv'],
                key='questions_upload',
                help="JSON: {topic: {difficulty: [questions]}} or CSV with columns"
            )
            
            if questions_file:
                if questions_file.name.endswith('.json'):
                    custom_questions = load_custom_questions_json(questions_file)
                else:
                    custom_questions = load_custom_questions_csv(questions_file)
                
                if custom_questions:
                    if st.button("✅ Use Custom Questions"):
                        st.session_state.custom_questions_db = custom_questions
                        st.rerun()
        
        # Export Data Section
        with st.expander("📥 Export Data", expanded=False):
            st.markdown("**Export Learner Profiles:**")
            if st.button("💾 Download Learners CSV"):
                csv_data = export_learner_data_csv(st.session_state.learners_data)
                st.download_button(
                    "⬇️ Download",
                    csv_data,
                    file_name="learners_data.csv",
                    mime="text/csv"
                )
            
            st.markdown("**Export Quiz History:**")
            if st.button("💾 Download Quiz History CSV"):
                csv_data = export_quiz_history_csv(st.session_state.learners_data)
                st.download_button(
                    "⬇️ Download",
                    csv_data,
                    file_name="quiz_history.csv",
                    mime="text/csv"
                )
        
        st.markdown("---")
        st.markdown("### 👤 Learner Dashboard")
        
        learner_names = list(st.session_state.learners_data.keys())
        selected_learner = st.selectbox(
            "Select Learner Profile",
            learner_names,
            key="learner_selector"
        )
        
        if selected_learner:
            st.session_state.current_learner = selected_learner
            learner_data = st.session_state.learners_data[selected_learner]
            
            classification = learner_data['classification']
            
            if classification == "Struggling Learner":
                badge_color = "#fc8181"
                icon = "🆘"
            elif classification == "Advanced Learner":
                badge_color = "#48bb78"
                icon = "🌟"
            else:
                badge_color = "#4299e1"
                icon = "📊"
            
            st.markdown(f"""
                <div style='background: {badge_color}; padding: 15px; border-radius: 10px; 
                            color: white; text-align: center; margin: 15px 0; font-weight: 700;'>
                    {icon} {classification}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Quick Stats")
            if learner_data['quiz_history']:
                latest_quiz = learner_data['quiz_history'][-1]
                st.metric("Latest Accuracy", f"{latest_quiz['accuracy']}%")
                st.metric("Engagement Score", f"{learner_data['engagement_score']}/100")
                st.metric("Learning Pace", learner_data['learning_pace'].title())
            
            st.markdown("### 🎯 AI Insights")
            adaptations = adapt_content(learner_data)
            st.info(f"**Recommended Difficulty:** {adaptations['difficulty'].title()}")
            
            if adaptations['enable_challenge_mode']:
                st.markdown('<div class="challenge-badge">🏆 Challenge Mode Active</div>', unsafe_allow_html=True)
            
            if adaptations['revision_needed']:
                st.warning("⚠️ Revision recommended")
            
            if st.session_state.using_custom_data:
                st.success("✨ Using Custom Dataset")
            
            if st.session_state.custom_questions_db:
                st.success("✨ Using Custom Questions")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📚 Learning Dashboard",
        "✏️ Take Quiz",
        "📊 Performance Analytics",
        "💡 Recommendations",
        "🤖 AI Tutor Chat",
        "📖 Dataset Guide"
    ])
    
    # TAB 1: Learning Dashboard
    with tab1:
        if st.session_state.current_learner:
            learner_data = st.session_state.learners_data[st.session_state.current_learner]
            
            st.markdown(f"## Welcome, {st.session_state.current_learner}! 👋")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if learner_data['quiz_history']:
                    avg_accuracy = np.mean([q['accuracy'] for q in learner_data['quiz_history']])
                    st.metric("Average Accuracy", f"{avg_accuracy:.1f}%", f"+{avg_accuracy-50:.1f}%")
                else:
                    st.metric("Average Accuracy", "N/A")
            
            with col2:
                total_quizzes = len(learner_data['quiz_history'])
                st.metric("Quizzes Completed", total_quizzes)
            
            with col3:
                st.metric("Engagement", f"{learner_data['engagement_score']}/100")
            
            with col4:
                if learner_data['quiz_history']:
                    total_hints = sum([q['hints_used'] for q in learner_data['quiz_history']])
                    st.metric("Total Hints Used", total_hints)
                else:
                    st.metric("Total Hints Used", "0")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💪 Strengths")
                for strength in learner_data['strengths']:
                    st.success(f"✓ {strength}")
            
            with col2:
                st.markdown("### 🎯 Areas for Improvement")
                for weakness in learner_data['weaknesses']:
                    st.warning(f"→ {weakness}")
            
            st.markdown("---")
            
            if learner_data['quiz_history']:
                st.markdown("### 📈 Performance Trends")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    perf_chart = create_performance_chart(learner_data)
                    if perf_chart:
                        st.plotly_chart(perf_chart, use_container_width=True, key="dashboard_perf")
                
                with col2:
                    topic_chart = create_topic_heatmap(learner_data)
                    if topic_chart:
                        st.plotly_chart(topic_chart, use_container_width=True, key="dashboard_topic")
        else:
            st.info("👈 Please select a learner profile from the sidebar to begin")
    
    # TAB 2: Take Quiz
    with tab2:
        if not st.session_state.current_learner:
            st.warning("Please select a learner profile first")
        else:
            learner_data = st.session_state.learners_data[st.session_state.current_learner]
            
            st.markdown("## 📝 Adaptive Quiz System")
            
            if not st.session_state.quiz_started:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get available topics
                    if st.session_state.custom_questions_db:
                        available_topics = list(st.session_state.custom_questions_db.keys())
                    else:
                        available_topics = ["Algebra", "Geometry", "Statistics", "Calculus"]
                    
                    topic = st.selectbox("Select Topic", available_topics)
                
                with col2:
                    recommended_difficulty = predict_next_difficulty(learner_data)
                    difficulty = st.selectbox(
                        "Select Difficulty",
                        ["easy", "medium", "hard"],
                        index=["easy", "medium", "hard"].index(recommended_difficulty)
                    )
                    
                    st.info(f"💡 AI Recommends: **{recommended_difficulty.title()}** based on your performance")
                
                if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
                    questions = get_quiz_questions(topic, difficulty)
                    if questions:
                        st.session_state.current_quiz = {
                            "topic": topic,
                            "difficulty": difficulty,
                            "questions": questions,
                            "start_time": datetime.now(),
                            "answers": [],
                            "times": [],
                            "hints_used": 0
                        }
                        st.session_state.current_question_index = 0
                        st.session_state.quiz_started = True
                        st.session_state.feedback_messages = []
                        st.rerun()
                    else:
                        st.error("No questions available for this topic/difficulty combination")
            
            else:
                quiz = st.session_state.current_quiz
                q_index = st.session_state.current_question_index
                
                if q_index < len(quiz['questions']):
                    question = quiz['questions'][q_index]
                    
                    progress = (q_index) / len(quiz['questions'])
                    st.progress(progress)
                    st.markdown(f"**Question {q_index + 1} of {len(quiz['questions'])}**")
                    
                    st.markdown(f"### {question['question']}")
                    
                    answer = st.radio(
                        "Select your answer:",
                        range(len(question['options'])),
                        format_func=lambda x: question['options'][x],
                        key=f"q_{q_index}"
                    )
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        if st.button("💡 Get Hint", key=f"hint_{q_index}"):
                            st.session_state.current_quiz['hints_used'] += 1
                            st.info(f"**Hint:** {question['hint']}")
                    
                    with col2:
                        if st.button("✅ Submit Answer", type="primary", key=f"submit_{q_index}"):
                            question_time = (datetime.now() - quiz['start_time']).seconds
                            quiz['answers'].append(answer)
                            quiz['times'].append(question_time)
                            
                            is_correct = answer == question['correct']
                            feedback_msg, feedback_type = generate_personalized_feedback(
                                question, answer, question['correct'], learner_data['classification']
                            )
                            
                            if is_correct:
                                st.success(feedback_msg)
                            else:
                                st.error(feedback_msg)
                                st.info(f"**Explanation:** {question['explanation']}")
                            
                            st.session_state.current_question_index += 1
                            quiz['start_time'] = datetime.now()
                            
                            if st.session_state.current_question_index >= len(quiz['questions']):
                                st.balloons()
                            
                            st.rerun()
                    
                    with col3:
                        if st.button("❌ Exit", key=f"exit_{q_index}"):
                            st.session_state.quiz_started = False
                            st.session_state.current_quiz = None
                            st.rerun()
                
                else:
                    st.markdown("## 🎉 Quiz Completed!")
                    
                    correct_answers = sum([1 for i, ans in enumerate(quiz['answers']) 
                                          if ans == quiz['questions'][i]['correct']])
                    accuracy = (correct_answers / len(quiz['questions'])) * 100
                    avg_time = np.mean(quiz['times']) if quiz['times'] else 0
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Score", f"{correct_answers}/{len(quiz['questions'])}")
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    with col3:
                        st.metric("Avg Time", f"{avg_time:.1f}s")
                    
                    new_quiz_record = {
                        "accuracy": accuracy,
                        "avg_time": avg_time,
                        "hints_used": quiz['hints_used'],
                        "retries": 0,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "topic": quiz['topic']
                    }
                    
                    learner_data['quiz_history'].append(new_quiz_record)
                    learner_data['classification'] = classify_learner(learner_data)
                    
                    if accuracy >= 90:
                        st.success("🌟 Outstanding performance! You're mastering this topic!")
                    elif accuracy >= 70:
                        st.success("👍 Good job! Keep practicing to reach mastery.")
                    else:
                        st.warning("💪 Don't worry! Review the material and try again. You've got this!")
                    
                    st.markdown("### 📚 What's Next?")
                    recommendations = generate_recommendations(learner_data, quiz['topic'])
                    
                    for rec in recommendations[:2]:
                        st.markdown(f"""
                            <div class="recommendation-card">
                                <strong>{rec['type']}</strong><br>
                                <span style='font-size: 1.1em;'>{rec['title']}</span><br>
                                <span style='font-size: 0.9em; color: #555;'>{rec['description']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("🔄 Take Another Quiz", type="primary"):
                        st.session_state.quiz_started = False
                        st.session_state.current_quiz = None
                        st.rerun()
    
    # TAB 3: Performance Analytics (ENHANCED)
    with tab3:
        if not st.session_state.current_learner:
            st.warning("Please select a learner profile first")
        else:
            learner_data = st.session_state.learners_data[st.session_state.current_learner]
            
            st.markdown("## 📊 Comprehensive Performance Analytics")
            
            if not learner_data['quiz_history']:
                st.info("No quiz data available yet. Take a quiz to see your analytics!")
            else:
                st.markdown("### 🎯 Overall Performance")
                
                col1, col2, col3 = st.columns(3)
                
                avg_accuracy = np.mean([q['accuracy'] for q in learner_data['quiz_history']])
                avg_time = np.mean([q['avg_time'] for q in learner_data['quiz_history']])
                
                with col1:
                    fig = create_metrics_gauge(avg_accuracy, "Average Accuracy")
                    st.plotly_chart(fig, use_container_width=True, key="analytics_gauge_accuracy")
                
                with col2:
                    fig = create_metrics_gauge(learner_data['engagement_score'], "Engagement Score")
                    st.plotly_chart(fig, use_container_width=True, key="analytics_gauge_engagement")
                
                with col3:
                    time_score = max(0, 100 - avg_time)
                    fig = create_metrics_gauge(time_score, "Speed Score")
                    st.plotly_chart(fig, use_container_width=True, key="analytics_gauge_speed")
                
                st.markdown("---")
                
                # Row 1: Accuracy Trend and Topic Performance
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Accuracy Trend")
                    perf_chart = create_performance_chart(learner_data)
                    if perf_chart:
                        st.plotly_chart(perf_chart, use_container_width=True, key="analytics_perf_trend")
                
                with col2:
                    st.markdown("### 📊 Topic Performance")
                    topic_chart = create_topic_heatmap(learner_data)
                    if topic_chart:
                        st.plotly_chart(topic_chart, use_container_width=True, key="analytics_topic_heatmap")
                
                st.markdown("---")
                
                # Row 2: NEW CHARTS - Time vs Accuracy and Improvement Trajectory
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ⚡ Time vs Accuracy Analysis")
                    scatter_chart = create_time_vs_accuracy_scatter(learner_data)
                    if scatter_chart:
                        st.plotly_chart(scatter_chart, use_container_width=True, key="analytics_scatter")
                
                with col2:
                    st.markdown("### 📈 Learning Progress Trajectory")
                    trajectory_chart = create_improvement_trajectory(learner_data)
                    if trajectory_chart:
                        st.plotly_chart(trajectory_chart, use_container_width=True, key="analytics_trajectory")
                
                st.markdown("---")
                
                # Row 3: NEW CHARTS - Hints Usage and Score Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💡 Hints Usage Pattern")
                    hints_chart = create_hints_usage_chart(learner_data)
                    if hints_chart:
                        st.plotly_chart(hints_chart, use_container_width=True, key="analytics_hints")
                
                with col2:
                    st.markdown("### 📊 Score Distribution")
                    dist_chart = create_performance_distribution(learner_data)
                    if dist_chart:
                        st.plotly_chart(dist_chart, use_container_width=True, key="analytics_distribution")
                
                st.markdown("---")
                
                st.markdown("### 📋 Quiz History")
                df = pd.DataFrame(learner_data['quiz_history'])
                st.dataframe(
                    df[['date', 'topic', 'accuracy', 'avg_time', 'hints_used']].sort_values('date', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("### 🤖 ML-Based Learner Clustering")
                clusters = cluster_learners(st.session_state.learners_data)
                
                if clusters:
                    if st.session_state.current_learner in clusters:
                        cluster_id = clusters[st.session_state.current_learner]
                        st.info(f"You are in **Cluster {cluster_id}** based on your performance patterns")
                        
                        similar_learners = [name for name, cid in clusters.items() 
                                          if cid == cluster_id and name != st.session_state.current_learner]
                        
                        if similar_learners:
                            st.write("**Similar learners in your cluster:**")
                            st.write(", ".join(similar_learners))
    
    # TAB 4: Recommendations
    with tab4:
        if not st.session_state.current_learner:
            st.warning("Please select a learner profile first")
        else:
            learner_data = st.session_state.learners_data[st.session_state.current_learner]
            
            st.markdown("## 💡 Personalized Learning Recommendations")
            
            latest_topic = learner_data['quiz_history'][-1]['topic'] if learner_data['quiz_history'] else "Algebra"
            
            recommendations = generate_recommendations(learner_data, latest_topic)
            
            st.markdown(f"### Based on your **{learner_data['classification']}** profile")
            
            for i, rec in enumerate(recommendations):
                with st.expander(f"{rec['type']}: {rec['title']}", expanded=(i < 2)):
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Priority:** {rec['priority']}")
                    
                    if rec['priority'] == "High":
                        st.markdown("🔥 **Highly Recommended for You**")
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Start", key=f"rec_{i}"):
                            st.success(f"Starting: {rec['title']}")
                    with col2:
                        if st.button("Save for Later", key=f"save_{i}"):
                            st.info("Saved to your learning path!")
            
            st.markdown("---")
            st.markdown("### 🎓 Why These Recommendations?")
            
            adaptations = adapt_content(learner_data)
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4>🤖 AI Adaptation Logic</h4>
                    <ul>
                        <li><strong>Recommended Difficulty:</strong> {adaptations['difficulty'].title()}</li>
                        <li><strong>Content Format:</strong> {adaptations['content_format'].title()}</li>
                        <li><strong>Hints Enabled:</strong> {'Yes' if adaptations['provide_hints'] else 'No'}</li>
                        <li><strong>Challenge Mode:</strong> {'Active' if adaptations['enable_challenge_mode'] else 'Not Active'}</li>
                        <li><strong>Revision Needed:</strong> {'Yes' if adaptations['revision_needed'] else 'No'}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    
    # TAB 5: AI Tutor Chat
    with tab5:
        if not st.session_state.current_learner:
            st.warning("Please select a learner profile first")
        else:
            learner_data = st.session_state.learners_data[st.session_state.current_learner]
            
            st.markdown("## 🤖 AI Tutor Assistant")
            
            st.info("💬 Ask me anything about your learning journey! I can help with hints, explanations, and study strategies.")
            
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
            
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            user_input = st.chat_input("Type your question here...")
            
            if user_input:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                
                classification = learner_data['classification']
                
                responses = {
                    "help": f"I'm here to help! As a {classification}, I recommend focusing on {'visual explanations and guided practice' if classification == 'Struggling Learner' else 'consistent practice' if classification == 'Average Learner' else 'challenging problems and advanced topics'}.",
                    "hint": "Let me know which question you need help with, and I'll provide a targeted hint without giving away the answer!",
                    "strategy": f"Based on your profile, try {'breaking problems into smaller steps and using visual aids' if classification == 'Struggling Learner' else 'mixed practice and regular review' if classification == 'Average Learner' else 'tackling advanced problems and teaching others'}.",
                    "default": "That's a great question! I'm here to support your learning. Could you be more specific about what you'd like help with?"
                }
                
                response = responses["default"]
                user_lower = user_input.lower()
                
                if "help" in user_lower or "how" in user_lower:
                    response = responses["help"]
                elif "hint" in user_lower or "clue" in user_lower:
                    response = responses["hint"]
                elif "strategy" in user_lower or "study" in user_lower or "learn" in user_lower:
                    response = responses["strategy"]
                elif "performance" in user_lower or "progress" in user_lower:
                    if learner_data['quiz_history']:
                        avg_acc = np.mean([q['accuracy'] for q in learner_data['quiz_history']])
                        response = f"Your current average accuracy is {avg_acc:.1f}%. {'Great progress!' if avg_acc >= 70 else 'Keep working - improvement takes time!'}"
                
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            st.markdown("### 🎯 Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 My Progress Summary", use_container_width=True):
                    if learner_data['quiz_history']:
                        avg_acc = np.mean([q['accuracy'] for q in learner_data['quiz_history']])
                        summary = f"You've completed {len(learner_data['quiz_history'])} quizzes with an average accuracy of {avg_acc:.1f}%. Keep up the great work! 🌟"
                    else:
                        summary = "You haven't taken any quizzes yet. Start your learning journey today! 🚀"
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": summary})
                    st.rerun()
            
            with col2:
                if st.button("💡 Study Tips", use_container_width=True):
                    tips = {
                        "Struggling Learner": "Try breaking down complex problems into smaller steps. Visual aids and practice problems with immediate feedback can really help!",
                        "Average Learner": "Mix up your practice with different difficulty levels. Review regularly and don't hesitate to challenge yourself!",
                        "Advanced Learner": "Keep yourself challenged with advanced problems. Consider teaching others to deepen your understanding!"
                    }
                    tip = tips[learner_data['classification']]
                    st.session_state.chat_messages.append({"role": "assistant", "content": tip})
                    st.rerun()
            
            with col3:
                if st.button("🎯 Next Steps", use_container_width=True):
                    adaptations = adapt_content(learner_data)
                    next_steps = f"I recommend trying a **{adaptations['difficulty']}** difficulty quiz next. "
                    
                    if adaptations['revision_needed']:
                        next_steps += "Also, some revision would be beneficial before moving forward."
                    elif adaptations['enable_challenge_mode']:
                        next_steps += "You're ready for challenge mode - let's push your limits!"
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": next_steps})
                    st.rerun()
    
    # TAB 6: Dataset Guide
    with tab6:
        st.markdown("## 📖 Custom Dataset Format Guide")
        
        st.markdown("""
        This guide explains how to format your custom datasets for the Smart Learning System.
        """)
        
        # Learner Profiles CSV
        st.markdown("### 1️⃣ Learner Profiles (CSV Format)")
        
        st.markdown("**Required Columns:**")
        st.code("""
learner_name,learner_id,engagement_score,learning_pace,classification,strengths,weaknesses
John Doe,L001,75,moderate,Average Learner,"Problem Solving,Consistency","Time Management,Complex Concepts"
Jane Smith,L002,92,fast,Advanced Learner,"Quick Learning,Self-Directed","Patience with Basics"
        """)
        
        st.markdown("""
        - **learner_name**: Student's full name
        - **learner_id**: Unique identifier (e.g., L001)
        - **engagement_score**: 0-100 score
        - **learning_pace**: slow, moderate, or fast
        - **classification**: Struggling Learner, Average Learner, or Advanced Learner
        - **strengths**: Comma-separated list (in quotes)
        - **weaknesses**: Comma-separated list (in quotes)
        """)
        
        # Quiz History CSV
        st.markdown("### 2️⃣ Quiz History (CSV Format)")
        
        st.markdown("**Required Columns:**")
        st.code("""
learner_name,accuracy,avg_time,hints_used,retries,date,topic
John Doe,75.5,45.2,2,0,2026-01-10,Algebra
John Doe,82.0,38.5,1,0,2026-01-12,Geometry
Jane Smith,95.0,25.0,0,0,2026-01-10,Algebra
        """)
        
        st.markdown("""
        - **learner_name**: Must match name in learner profiles
        - **accuracy**: 0-100 percentage score
        - **avg_time**: Average seconds per question
        - **hints_used**: Number of hints used in quiz
        - **retries**: Number of retry attempts
        - **date**: YYYY-MM-DD format
        - **topic**: Subject area (Algebra, Geometry, etc.)
        """)
        
        # Custom Questions CSV
        st.markdown("### 3️⃣ Custom Questions (CSV Format)")
        
        st.markdown("**Required Columns:**")
        st.code("""
topic,difficulty,question,option1,option2,option3,option4,correct_index,hint,explanation
Algebra,easy,"What is 2 + 2?",3,4,5,6,1,"Add the numbers","2 + 2 equals 4"
Geometry,medium,"What is the area of a circle with radius 5?","25π","10π","5π","50π",0,"Use formula A = πr²","A = π × 5² = 25π"
        """)
        
        st.markdown("""
        - **topic**: Subject area (can be custom)
        - **difficulty**: easy, medium, or hard
        - **question**: The question text
        - **option1-4**: Four multiple choice options
        - **correct_index**: 0-3 (which option is correct, 0-indexed)
        - **hint**: Helpful hint without giving answer
        - **explanation**: Full explanation of the answer
        """)
        
        # Custom Questions JSON
        st.markdown("### 4️⃣ Custom Questions (JSON Format)")
        
        st.markdown("**Structure:**")
        st.code("""
{
  "Algebra": {
    "easy": [
      {
        "question": "Solve for x: 2x + 5 = 13",
        "options": ["x = 3", "x = 4", "x = 5", "x = 6"],
        "correct": 1,
        "hint": "Subtract 5 from both sides first",
        "explanation": "First subtract 5: 2x = 8, then divide: x = 4"
      }
    ],
    "medium": [ ... ],
    "hard": [ ... ]
  },
  "Geometry": { ... }
}
        """, language="json")
        
        # Download Templates
        st.markdown("### 📥 Download Templates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            template_learners = """learner_name,learner_id,engagement_score,learning_pace,classification,strengths,weaknesses
John Doe,L001,75,moderate,Average Learner,"Problem Solving,Consistency","Time Management"
Jane Smith,L002,92,fast,Advanced Learner,"Quick Learning","Patience"
"""
            st.download_button(
                "⬇️ Learners Template",
                template_learners,
                file_name="learners_template.csv",
                mime="text/csv"
            )
        
        with col2:
            template_quiz = """learner_name,accuracy,avg_time,hints_used,retries,date,topic
John Doe,75.5,45.2,2,0,2026-01-10,Algebra
Jane Smith,95.0,25.0,0,0,2026-01-10,Algebra
"""
            st.download_button(
                "⬇️ Quiz History Template",
                template_quiz,
                file_name="quiz_history_template.csv",
                mime="text/csv"
            )
        
        with col3:
            template_questions = """topic,difficulty,question,option1,option2,option3,option4,correct_index,hint,explanation
Algebra,easy,"What is 2 + 2?",3,4,5,6,1,"Add the numbers","2 + 2 equals 4"
"""
            st.download_button(
                "⬇️ Questions Template",
                template_questions,
                file_name="questions_template.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips for Best Results
        
        1. **Consistent Naming**: Ensure learner names match exactly across files
        2. **Date Format**: Always use YYYY-MM-DD format for dates
        3. **Encoding**: Save CSV files with UTF-8 encoding
        4. **Quotes**: Use quotes around text with commas (especially strengths/weaknesses)
        5. **Test Small**: Start with 2-3 rows to test your format
        6. **Valid Options**: Ensure correct_index is 0-3 for multiple choice
        7. **Custom Topics**: You can create any topic names you want
        
        ### 🔧 Troubleshooting
        
        - **"Column not found"**: Check spelling of column names
        - **"Invalid format"**: Ensure CSV is properly formatted
        - **"No data loaded"**: Check file encoding (use UTF-8)
        - **Questions not appearing**: Verify topic/difficulty match your selections
        """)

if __name__ == "__main__":
    main()