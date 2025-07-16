# 🌿 Green Guide: Waste Segregation & Reporting System

Green Guide is a web-based platform promoting **sustainable waste management** by empowering **citizens, government officials, and administrators**. With an intuitive UI, real-time waste detection using **YOLOv8**, and interactive community features, Green Guide aims to foster eco-conscious behavior through technology.

---

## 🚀 Key Features

- 🔐 **User Authentication**  
  Role-based access: *Citizen*, *Government Official*, *Admin*

- 🗑️ **Waste Reporting**  
  Upload images with location and description to report improper waste disposal

- 📚 **Waste Segregation Guide**  
  Educational resources for correct waste disposal and segregation

- 📸 **Real-Time Image Processing**  
  YOLOv8-based detection and classification of waste via image/webcam input

- 🏛️ **Government Dashboard**  
  Officials can review and manage report statuses (Pending, Approved, Not Approved, Done)

- 💬 **Community Discussion Forum**  
  Post updates, comment, like, and share environmental schemes

- 👤 **User Profiles**  
  Customizable profiles with full name, bio, and profile picture

- ⭐ **Review System**  
  Users can rate and review the platform (moderated by Admin)

- ⚙️ **Admin Dashboard**  
  Centralized content monitoring and moderation tools

- 📱 **Responsive Design**  
  Fully mobile-friendly with intuitive templates

---

## 🛠 Tech Stack

| Layer     | Tech                                           |
|-----------|------------------------------------------------|
| Backend   | Flask (Python)                                 |
| Frontend  | HTML, CSS, JavaScript (Jinja2 templating)      |
| Database  | MySQL                                          |
| ML Model  | YOLOv8 (via Ultralytics)                       |
| Libraries | OpenCV, Pillow, Flask-Login, Werkzeug          |
| Environment | Python 3.8+, MySQL 8.0+                      |
| Deployment | Configurable via `.env` variables             |

---

## ⚙️ Installation & Setup

### ✅ Prerequisites

- Python 3.8 or higher  
- MySQL 8.0 or higher  
- `pip` (Python package manager)  
- Git  

### 🔧 Setup Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/yuvrajbhatkariya/GreenGuide.git
cd GreenGuide
```

### 🔧 Create & Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment

# On macOS/Linux:
source venv/bin/activate

# On Windows CMD:
venv\Scripts\activate

# On Windows PowerShell:
venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
SECRET_KEY=your-secret-key
DB_HOST=localhost
DB_USER=your-mysql-username
DB_PASSWORD=your-mysql-password
DB_NAME=green_guide_db
```

#### 5. Add YOLOv8 Model Weights

Download and place `best.pt` in the `model/` directory.  
Ensure the following path is correctly referenced in the code:

```python
MODEL_PATH = "model/best.pt"
```

#### 6. Initialize the Database

Start the server to auto-create tables:

```bash
python app.py
```

Or open in browser:  
[`http://localhost:5001/init_db`](http://localhost:5001/init_db)

#### 7. Run the App

```bash
python app.py
```

Visit: [`http://localhost:5001`](http://localhost:5001)

---

## 📁 Project Structure

```
green-guide/
├── model/                  # YOLOv8 model weights
├── static/                 # Static assets (CSS, JS, uploads)
│   └── uploads/            # Uploaded user images
├── templates/              # HTML templates (Jinja2)
├── .env                    # Environment config
├── app.py                  # Main Flask app
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📌 How to Use

| Page/Route              | Purpose |
|-------------------------|---------|
| `/signup`, `/login`     | Create or log into an account |
| `/report`               | Report improper waste disposal |
| `/guide`                | Learn about waste segregation |
| `/government_reports`   | Officials manage reports |
| `/discussion`           | Join community discussions |
| `/profile`              | Edit your profile and photo |
| `/reviews`              | View and submit platform reviews |
| `/features_breakdown_dashboard` | Admin dashboard for content control |

---

## 🧬 Database Schema

| Table         | Description |
|---------------|-------------|
| `users`       | User info (username, email, role, etc.) |
| `reports`     | Waste reports (location, image, status) |
| `subscriptions` | Email newsletter subscriptions |
| `posts`       | Discussion posts |
| `comments`    | Comments on posts |
| `reviews`     | Platform reviews and ratings |

---

## 🔒 Security

- Passwords hashed via `Werkzeug`
- Role-based route access
- CSRF protection with Flask sessions
- UUID-based secure file uploads

---

## 🧰 Troubleshooting

| Issue                   | Solution |
|-------------------------|----------|
| YOLOv8 Model not found  | Ensure `best.pt` is in `model/` folder |
| Database errors         | Check MySQL credentials in `.env` and server status |
| Image processing fails  | Confirm OpenCV and Pillow are installed |
| Port already in use     | Change port in `app.run(port=5001)` |

---

## 🤝 Contributing

We welcome contributions from developers and environmentalists!

1. Fork the repo  
2. Create a feature branch: `git checkout -b feature/your-feature`  
3. Commit changes: `git commit -m "Add your feature"`  
4. Push to branch: `git push origin feature/your-feature`  
5. Open a Pull Request 🎉

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 📬 Contact

📨 For questions, suggestions, or support:  
Visit the [/contact](http://localhost:5001/contact) page  
or email: `support@example.com`

---

Let’s make the world cleaner, one report at a time 🌍♻️
