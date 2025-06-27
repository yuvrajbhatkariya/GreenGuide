import io
import os
import cv2
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
import mysql.connector
from mysql.connector import Error
import logging
import re
import uuid
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")  # Fallback for local testing

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

load_dotenv()

# MySQL configuration
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "raise_on_warnings": False
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        if conn.is_connected():
            logger.info("Successfully connected to MySQL database")
            return conn
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {str(e)}")
        raise

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            password VARCHAR(255) NOT NULL,
            user_type ENUM('citizen', 'government_official', 'admin') NOT NULL,
            full_name VARCHAR(255),
            profile_photo VARCHAR(255),
            bio TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            house_number VARCHAR(50),
            street_name VARCHAR(255),
            city VARCHAR(100),
            pin_code VARCHAR(20),
            country VARCHAR(100),
            description TEXT NOT NULL,
            image_path VARCHAR(255) NOT NULL,
            status ENUM('Pending', 'Approved', 'Not Approved', 'Done') DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS subscriptions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS posts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            username VARCHAR(255),
            content TEXT NOT NULL,
            image_path VARCHAR(255),
            link VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS comments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            post_id INT,
            user_id INT,
            username VARCHAR(255),
            content TEXT NOT NULL,
            parent_comment_id INT,
            likes INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (parent_comment_id) REFERENCES comments(id)
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS reviews (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            username VARCHAR(255),
            rating INT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        conn.commit()
        logger.info("Tables created or already exist")
    except Error as e:
        logger.error(f"Error creating table: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

# Initialize database only once (manually or via endpoint)
# Comment out init_db() here to avoid running on every start
# init_db()

class User(UserMixin):
    def __init__(self, id, username, user_type, email=None, full_name=None, profile_photo=None, bio=None):
        self.id = id
        self.username = username
        self.user_type = user_type
        self.email = email
        self.full_name = full_name
        self.profile_photo = profile_photo
        self.bio = bio

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, username, user_type, email, full_name, profile_photo, bio FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()
        if user:
            return User(user[0], user[1], user[2], user[3], user[4], user[5], user[6])
        return None
    except Error as e:
        logger.error(f"Error loading user: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

def generate_unique_username(email):
    base_username = re.sub(r'[^a-zA-Z0-9]', '', email.split('@')[0])[:20]
    username = base_username
    count = 1
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        while True:
            cursor.execute('SELECT username FROM users WHERE username = %s', (username,))
            if not cursor.fetchone():
                return username
            username = f"{base_username}{count}"
            count += 1
    except Error as e:
        logger.error(f"Error generating username: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

MODEL_PATH = "model/best.pt"
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model weights not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Successfully loaded YOLOv8 model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLOv8 model: {str(e)}")
    raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")

def process_frame(frame):
    try:
        results = model.predict(source=frame, conf=0.45, iou=0.45, device='cpu')
        annotated_frame = results[0].plot()
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = r.names[cls]
                detections.append({"class": class_name, "confidence": conf})
        return annotated_frame, detections
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT r.id, r.user_id, r.username, r.rating, r.content, r.created_at, 
                      u.profile_photo
               FROM reviews r
               JOIN users u ON r.user_id = u.id
               ORDER BY r.created_at DESC
               LIMIT 5'''
        )
        recent_reviews = cursor.fetchall()
        
        has_reviewed = False
        if current_user.is_authenticated:
            cursor.execute('SELECT id FROM reviews WHERE user_id = %s', (current_user.id,))
            has_reviewed = cursor.fetchone() is not None
            
    except Error as e:
        logger.error(f"Error fetching reviews: {str(e)}")
        recent_reviews = []
        has_reviewed = False
    finally:
        cursor.close()
        conn.close()
    
    return render_template('index.html', recent_reviews=recent_reviews, has_reviewed=has_reviewed)

@app.route('/init_db')
@login_required
def initialize_db():
    if current_user.user_type != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
    init_db()
    flash('Database initialized successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/webcam')
@login_required
def webcam():
    return render_template('webcam.html')

@app.route('/process_frame', methods=['POST'])
@login_required
def process_frame_endpoint():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        npimg = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        processed_frame, detections = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'image': f'data:image/jpeg;base64,{encoded_image}',
            'detections': detections
        })
    except Exception as e:
        logger.error(f"Error in process_frame_endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.', 'danger')
            return redirect(request.url)
        try:
            img = Image.open(file).convert('RGB')
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_frame, detections = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return render_template('upload.html', image=f'data:image/jpeg;base64,{encoded_image}', detections=detections)
        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(request.url)
    return render_template('upload.html', image=None, detections=None)

@app.route('/report', methods=['GET', 'POST'])
@login_required
def report():
    if current_user.user_type == 'government_official':
        return redirect(url_for('government_reports'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(request.url)
        file = request.files['file']
        house_number = request.form.get('house_number')
        street_name = request.form.get('street_name')
        city = request.form.get('city')
        pin_code = request.form.get('pin_code')
        country = request.form.get('country')
        description = request.form.get('description')
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.', 'danger')
            return redirect(request.url)
        if not all([house_number, street_name, city, pin_code, country, description]):
            flash('All fields are required', 'danger')
            return redirect(request.url)
        
        try:
            img = Image.open(file).convert('RGB')
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_frame, detections = process_frame(frame)
            filename = f"{uuid.uuid4()}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(image_path, processed_frame)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO reports (user_id, house_number, street_name, city, pin_code, country, description, image_path, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                (current_user.id, house_number, street_name, city, pin_code, country, description, image_path, 'Pending')
            )
            conn.commit()
            _, buffer = cv2.imencode('.jpg', processed_frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            flash('Report submitted successfully!', 'success')
            return render_template('report.html', image=f'data:image/jpeg;base64,{encoded_image}', detections=detections)
        except Exception as e:
            flash(f'Error processing report: {str(e)}', 'danger')
            return redirect(request.url)
        finally:
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT id, house_number, street_name, city, pin_code, country, description, image_path, status, created_at
               FROM reports WHERE user_id = %s ORDER BY created_at DESC''',
            (current_user.id,)
        )
        reports = cursor.fetchall()
    except Error as e:
        logger.error(f"Error fetching user reports: {str(e)}")
        flash('Error fetching your reports', 'danger')
        reports = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('report.html', image=None, detections=None, reports=reports)

@app.route('/government_reports', methods=['GET', 'POST'])
@login_required
def government_reports():
    if current_user.user_type != 'government_official':
        flash('Access denied. Government officials only.', 'danger')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        report_id = request.form.get('report_id')
        new_status = request.form.get('status')
        if not report_id or new_status not in ['Approved', 'Not Approved', 'Done']:
            flash('Invalid request', 'danger')
            return redirect(request.url)
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE reports SET status = %s WHERE id = %s',
                (new_status, report_id)
            )
            conn.commit()
            flash(f'Report status updated to {new_status}', 'success')
        except Error as e:
            logger.error(f"Error updating report status: {str(e)}")
            flash('Error updating report status', 'danger')
        finally:
            cursor.close()
            conn.close()
        return redirect(request.url)
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT r.id, r.user_id, u.username, r.house_number, r.street_name, r.city, r.pin_code, r.country,
                      r.description, r.image_path, r.status, r.created_at
               FROM reports r JOIN users u ON r.user_id = u.id
               ORDER BY r.created_at DESC'''
        )
        reports = cursor.fetchall()
    except Error as e:
        logger.error(f"Error fetching reports: {str(e)}")
        flash('Error fetching reports', 'danger')
        reports = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('government_reports.html', reports=reports)

@app.route('/discussion', methods=['GET', 'POST'])
@login_required
def discussion():
    if request.method == 'POST':
        content = request.form.get('content')
        link = request.form.get('link')
        file = request.files.get('file')
        
        if not content:
            flash('Content is required', 'danger')
            return redirect(request.url)
        
        if link and not re.match(r'^https?://', link):
            flash('Invalid link format', 'danger')
            return redirect(request.url)
        
        image_path = None
        if file and file.filename:
            if not allowed_file(file.filename):
                flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.', 'danger')
                return redirect(request.url)
            try:
                img = Image.open(file).convert('RGB')
                frame = np.array(img)
                filename = f"post_{uuid.uuid4()}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO posts (user_id, username, content, image_path, link)
                   VALUES (%s, %s, %s, %s, %s)''',
                (current_user.id, current_user.username, content, image_path, link)
            )
            conn.commit()
            flash('Post created successfully!', 'success')
        except Error as e:
            logger.error(f"Error creating post: {str(e)}")
            flash('Error creating post', 'danger')
        finally:
            cursor.close()
            conn.close()
        return redirect(url_for('discussion'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT id, user_id, username, content, image_path, link, created_at
               FROM posts ORDER BY created_at DESC'''
        )
        posts = cursor.fetchall()
        
        cursor.execute(
            '''SELECT c.id, c.post_id, c.user_id, c.username, c.content, c.parent_comment_id, c.likes, c.created_at,
                      p.id AS post_id_check
               FROM comments c
               JOIN posts p ON c.post_id = p.id
               ORDER BY c.created_at DESC'''
        )
        comments = cursor.fetchall()
        comments_by_post = {}
        for comment in comments:
            post_id = comment['post_id']
            if post_id not in comments_by_post:
                comments_by_post[post_id] = []
            comments_by_post[post_id].append(comment)
    except Error as e:
        logger.error(f"Error fetching posts or comments: {str(e)}")
        flash('Error fetching posts or comments', 'danger')
        posts = []
        comments_by_post = {}
    finally:
        cursor.close()
        conn.close()
    
    return render_template('discussion.html', posts=posts, comments_by_post=comments_by_post)

@app.route('/discussion/comment', methods=['POST'])
@login_required
def add_comment():
    post_id = request.form.get('post_id')
    content = request.form.get('content')
    parent_comment_id = request.form.get('parent_comment_id')
    
    if not content or not post_id:
        flash('Content and post ID are required', 'danger')
        return redirect(url_for('discussion'))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO comments (post_id, user_id, username, content, parent_comment_id) VALUES (%s, %s, %s, %s, %s)',
            (post_id, current_user.id, current_user.username, content, parent_comment_id or None)
        )
        conn.commit()
        flash('Comment added successfully!', 'success')
    except Error as e:
        logger.error(f"Error adding comment: {str(e)}")
        flash('Error adding comment', 'danger')
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('discussion'))

@app.route('/discussion/like/<int:comment_id>', methods=['POST'])
@login_required
def like_comment(comment_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE comments SET likes = likes + 1 WHERE id = %s',
            (comment_id,)
        )
        conn.commit()
        flash('Comment liked!', 'success')
    except Error as e:
        logger.error(f"Error liking comment: {str(e)}")
        flash('Error liking comment', 'danger')
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('discussion'))

@app.route('/discussion/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT user_id FROM comments WHERE id = %s', (comment_id,))
        comment = cursor.fetchone()
        if not comment:
            flash('Comment not found', 'danger')
            return redirect(url_for('discussion'))
        
        if current_user.id != comment[0] and current_user.user_type != 'admin':
            flash('You can only delete your own comments', 'danger')
            return redirect(url_for('discussion'))
        
        cursor.execute('DELETE FROM comments WHERE id = %s', (comment_id,))
        conn.commit()
        flash('Comment deleted successfully!', 'success')
    except Error as e:
        logger.error(f"Error deleting comment: {str(e)}")
        flash('Error deleting comment', 'danger')
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('discussion'))

@app.route('/discussion/delete/<int:post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT user_id, image_path FROM posts WHERE id = %s', (post_id,))
        post = cursor.fetchone()
        if not post:
            flash('Post not found', 'danger')
            return redirect(url_for('discussion'))
        
        if current_user.id != post[0] and current_user.user_type != 'admin':
            flash('You can only delete your own posts', 'danger')
            return redirect(url_for('discussion'))
        
        if post[1] and os.path.exists(post[1]):
            os.remove(post[1])
        
        cursor.execute('DELETE FROM posts WHERE id = %s', (post_id,))
        conn.commit()
        flash('Post deleted successfully!', 'success')
    except Error as e:
        logger.error(f"Error deleting post: {str(e)}")
        flash('Error deleting post', 'danger')
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('discussion'))

@app.route('/features_breakdown_dashboard')
@login_required
def features_breakdown_dashboard():
    if current_user.user_type != 'admin':
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT id, user_id, username, content, image_path, link, created_at
               FROM posts ORDER BY created_at DESC'''
        )
        posts = cursor.fetchall()
    except Error as e:
        logger.error(f"Error fetching posts: {str(e)}")
        posts = []
    finally:
        cursor.close()
        conn.close()
    
    return render_template('features_breakdown.html', posts=posts)

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, username, password, user_type FROM users WHERE email = %s', (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user[2], password):
                login_user(User(user[0], user[1], user[3]))
                flash('Logged in successfully!', 'success')
                return redirect(url_for('index'))
            flash('Invalid email or password', 'danger')
        except Error as e:
            logger.error(f"Error during login: {str(e)}")
            flash('An error occurred. Please try again.', 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        username = generate_unique_username(email)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (username, email, password, user_type) VALUES (%s, %s, %s, %s)',
                (username, email, generate_password_hash(password), user_type)
            )
            conn.commit()
            flash(f'Account created! Username: {username}', 'success')
            return redirect(url_for('login'))
        except Error as e:
            logger.error(f"Error during signup: {str(e)}")
            flash('Email already registered', 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/features_breakdown', methods=['GET', 'POST'])
def features_breakdown():
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id, username, user_type FROM users WHERE email = %s AND user_type = %s', (email, 'admin'))
            user = cursor.fetchone()
            if user:
                login_user(User(user[0], user[1], user[2]))
                flash('Access granted to Features Breakdown.', 'success')
                return redirect(url_for('features_breakdown_dashboard'))
            flash('Access denied. Admins only.', 'danger')
        except Error as e:
            logger.error(f"Error in features_breakdown: {str(e)}")
            flash('An error occurred. Please try again.', 'danger')
        finally:
            cursor.close()
            conn.close()
    return render_template('login.html', features_breakdown=True)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        bio = request.form.get('bio')
        profile_photo = request.files.get('profile_photo')
        
        db_image_path = current_user.profile_photo
        
        if profile_photo and profile_photo.filename:
            if not allowed_file(profile_photo.filename):
                flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.', 'danger')
                return redirect(url_for('profile'))
            try:
                filename = f"profile_{uuid.uuid4()}.jpg"
                file_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img = Image.open(profile_photo).convert('RGB')
                img = img.resize((300, 300), Image.LANCZOS)
                img.save(file_save_path)
                db_image_path = f"uploads/{filename}"
                
                if current_user.profile_photo:
                    old_path = os.path.join('static', current_user.profile_photo)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        
            except Exception as e:
                logger.error(f"Error processing profile photo: {str(e)}")
                flash(f'Error processing profile photo: {str(e)}', 'danger')
                return redirect(url_for('profile'))

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''UPDATE users SET full_name = %s, bio = %s, profile_photo = %s WHERE id = %s''',
                (full_name, bio, db_image_path, current_user.id)
            )
            conn.commit()
            current_user.full_name = full_name
            current_user.bio = bio
            current_user.profile_photo = db_image_path
            flash('Profile updated successfully!', 'success')
        except Error as e:
            logger.error(f"Error updating profile: {str(e)}")
            flash('Error updating profile', 'danger')
        finally:
            cursor.close()
            conn.close()
        return redirect(url_for('profile'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT id, house_number, street_name, city, pin_code, country, description, image_path, status, created_at
               FROM reports WHERE user_id = %s ORDER BY created_at DESC''',
            (current_user.id,)
        )
        reports = cursor.fetchall()
        cursor.execute(
            '''SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END) as approved,
                SUM(CASE WHEN status = 'Not Approved' THEN 1 ELSE 0 END) as not_approved,
                SUM(CASE WHEN status = 'Done' THEN 1 ELSE 0 END) as done
               FROM reports WHERE user_id = %s''',
            (current_user.id,)
        )
        reports_summary = cursor.fetchone()
    except Error as e:
        logger.error(f"Error fetching profile data: {str(e)}")
        flash('Error fetching profile data', 'danger')
        reports = []
        reports_summary = {'total': 0, 'pending': 0, 'approved': 0, 'not_approved': 0, 'done': 0}
    finally:
        cursor.close()
        conn.close()

    return render_template('profile.html', user=current_user, reports=reports, reports_summary=reports_summary)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        email = request.form.get('email')
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE users SET email = %s WHERE id = %s', (email, current_user.id))
            conn.commit()
            flash('Settings updated successfully!', 'success')
        except Error as e:
            logger.error(f"Error updating settings: {str(e)}")
            flash('Error updating settings', 'danger')
        finally:
            cursor.close()
            conn.close()
        return redirect(url_for('settings'))
    return render_template('settings.html', user=current_user)

@app.route('/blog')
def blog():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute('SELECT id, title, content, created_at FROM posts ORDER BY created_at DESC')
        posts = cursor.fetchall()
    except Error as e:
        logger.error(f"Error fetching posts: {str(e)}")
        posts = []
    finally:
        cursor.close()
        conn.close()
    return render_template('blog.html', posts=posts)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/reviews', methods=['GET', 'POST'])
@login_required
def reviews():
    if request.method == 'POST':
        rating = request.form.get('rating')
        content = request.form.get('content')
        
        if not rating or not content:
            flash('Rating and content are required', 'danger')
            return redirect(url_for('reviews'))
        
        try:
            rating = int(rating)
            if rating < 1 or rating > 5:
                flash('Rating must be between 1 and 5', 'danger')
                return redirect(url_for('reviews'))
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO reviews (user_id, username, rating, content)
                   VALUES (%s, %s, %s, %s)''',
                (current_user.id, current_user.username, rating, content)
            )
            conn.commit()
            flash('Review submitted successfully!', 'success')
        except Error as e:
            logger.error(f"Error submitting review: {str(e)}")
            flash('Error submitting review', 'danger')
        finally:
            if 'conn' in locals():
                cursor.close()
                conn.close()
        return redirect(url_for('reviews'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            '''SELECT r.id, r.user_id, r.username, r.rating, r.content, r.created_at, 
                      u.profile_photo
               FROM reviews r
               JOIN users u ON r.user_id = u.id
               ORDER BY r.created_at DESC'''
        )
        all_reviews = cursor.fetchall()
        
        cursor.execute('SELECT AVG(rating) as avg_rating FROM reviews')
        avg_rating = cursor.fetchone()['avg_rating'] or 0
        
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as one_star,
                SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as two_star,
                SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as three_star,
                SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) as four_star,
                SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) as five_star
            FROM reviews
        ''')
        rating_dist = cursor.fetchone()
        
        cursor.execute('SELECT id FROM reviews WHERE user_id = %s', (current_user.id,))
        has_reviewed = cursor.fetchone() is not None
        
    except Error as e:
        logger.error(f"Error fetching reviews: {str(e)}")
        flash('Error fetching reviews', 'danger')
        all_reviews = []
        avg_rating = 0
        rating_dist = {'one_star': 0, 'two_star': 0, 'three_star': 0, 'four_star': 0, 'five_star': 0}
        has_reviewed = False
    finally:
        cursor.close()
        conn.close()
    
    return render_template('reviews.html', 
                         reviews=all_reviews,
                         avg_rating=float(avg_rating),
                         rating_dist=rating_dist,
                         has_reviewed=has_reviewed)

@app.route('/reviews/delete/<int:review_id>', methods=['POST'])
@login_required
def delete_review(review_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT user_id FROM reviews WHERE id = %s', (review_id,))
        review = cursor.fetchone()
        
        if not review:
            flash('Review not found', 'danger')
        elif current_user.id != review[0] and current_user.user_type != 'admin':
            flash('You can only delete your own reviews', 'danger')
        else:
            cursor.execute('DELETE FROM reviews WHERE id = %s', (review_id,))
            conn.commit()
            flash('Review deleted successfully', 'success')
    except Error as e:
        logger.error(f"Error deleting review: {str(e)}")
        flash('Error deleting review', 'danger')
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('reviews'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)