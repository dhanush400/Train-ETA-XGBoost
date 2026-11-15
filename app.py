
import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)

from werkzeug.security import generate_password_hash, check_password_hash


# --------------------------------------------------
# 1. PROJECT PATHS AND FLASK CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "models"

DB_PATH = Path(os.environ.get("DATABASE_PATH", str(BASE_DIR / "users.db")))

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "railpulse-local-development-key"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("RENDER") is not None
)


# --------------------------------------------------
# 2. DATABASE INITIALIZATION
# --------------------------------------------------

def get_db_connection():
    """Create a connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")

    return connection


def initialize_database():
    """Create the users table if it does not exist."""

    connection = get_db_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password TEXT
            )
        """)

        # Add missing columns if an older users table exists.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

        if "identifier" not in columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN identifier TEXT"
            )

        if "password_hash" not in columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN password_hash TEXT"
            )

        if "password" not in columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN password TEXT"
            )

        connection.commit()

    finally:
        connection.close()


# Create the database table when the application starts.
initialize_database()


# --------------------------------------------------
# 3. LOAD XGBOOST MODEL AND METRICS
# --------------------------------------------------

MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

model = None
model_error = None
metrics = {}

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model_error = (
            "Model file not found: models/xgboost_model.pkl"
        )

except Exception as exc:
    model_error = f"Could not load the prediction model: {exc}"


try:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

except Exception:
    metrics = {}


# Feature order must match the model training columns.
FEATURE_COLUMNS = [
    "distance_km",
    "scheduled_hour",
    "day_of_week",
    "month",
    "remaining_stops",
    "previous_delay_min",
    "weather_severity",
    "congestion_index",
    "train_type_code",
]


# --------------------------------------------------
# 4. LOGIN PROTECTION
# --------------------------------------------------

def login_required(function):
    """Prevent users from accessing the dashboard without login."""

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


# --------------------------------------------------
# 5. LOGIN PAGE
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":

        identifier = request.form.get(
            "identifier", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        if not identifier or not password:
            error = "Please enter your email and password."

        else:
            connection = get_db_connection()

            try:
                user = connection.execute(
                    """
                    SELECT id, identifier, password_hash, password
                    FROM users
                    WHERE LOWER(identifier) = ?
                    """,
                    (identifier,)
                ).fetchone()

            finally:
                connection.close()

            if user is None:
                error = "Account not found. Please register first."

            else:
                stored_hash = user["password_hash"] or ""
                stored_password = user["password"] or ""

                password_valid = False

                if stored_hash:
                    try:
                        password_valid = check_password_hash(
                            stored_hash, password
                        )
                    except (ValueError, TypeError):
                        password_valid = False

                # Backward compatibility with older plaintext records.
                if not password_valid and stored_password:
                    password_valid = (
                        stored_password == password
                    )

                if password_valid:

                    session.clear()

                    session["user_id"] = user["id"]
                    session["identifier"] = user["identifier"]

                    # Upgrade an old plaintext password to a hash.
                    if stored_password:
                        connection = get_db_connection()

                        try:
                            connection.execute(
                                """
                                UPDATE users
                                SET password_hash = ?, password = NULL
                                WHERE id = ?
                                """,
                                (
                                    generate_password_hash(password),
                                    user["id"]
                                )
                            )
                            connection.commit()

                        finally:
                            connection.close()

                    return redirect(url_for("home"))

                else:
                    error = "Incorrect password. Please try again."

    return render_template(
        "auth.html",
        error=error,
        mode="login"
    )


# --------------------------------------------------
# 6. REGISTRATION PAGE
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":

        identifier = request.form.get(
            "identifier", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not identifier or not password or not confirm_password:
            error = "Please fill in all the fields."

        elif password != confirm_password:
            error = "Passwords do not match."

        elif len(password) < 7:
            error = "Password must contain at least 7 characters."

        else:
            connection = get_db_connection()

            try:
                existing_user = connection.execute(
                    """
                    SELECT id FROM users
                    WHERE LOWER(identifier) = ?
                    """,
                    (identifier,)
                ).fetchone()

                if existing_user:
                    error = "This account already exists. Please login."

                else:
                    password_hash = generate_password_hash(password)

                    cursor = connection.execute(
                        """
                        INSERT INTO users
                        (identifier, password_hash, password)
                        VALUES (?, ?, NULL)
                        """,
                        (identifier, password_hash)
                    )

                    connection.commit()

                    # Automatically log the user in after registration.
                    session.clear()
                    session["user_id"] = cursor.lastrowid
                    session["identifier"] = identifier

                    return redirect(url_for("home"))

            except sqlite3.IntegrityError:
                error = "This account already exists. Please login."

            except sqlite3.Error:
                app.logger.exception("Registration database error")
                error = "Registration failed because of a database error."

            finally:
                connection.close()

    return render_template(
        "auth.html",
        error=error,
        mode="register"
    )


# --------------------------------------------------
# 7. DASHBOARD AND TRAIN ETA PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    result = None
    error = None

    if request.method == "POST":

        try:
            if model is None:
                raise RuntimeError(
                    model_error or "Prediction model is unavailable."
                )

            train_number = request.form.get(
                "train_number", ""
            ).strip()

            scheduled_input = request.form.get(
                "scheduled_arrival", ""
            ).strip()

            if not train_number:
                raise ValueError("Please enter the train number.")

            if not scheduled_input:
                raise ValueError(
                    "Please enter the scheduled arrival date and time."
                )

            # Accept datetime-local input and common date formats.
            try:
                scheduled_arrival = datetime.fromisoformat(
                    scheduled_input
                )
            except ValueError:
                scheduled_arrival = datetime.strptime(
                    scheduled_input,
                    "%Y-%m-%d %H:%M"
                )

            distance_km = float(request.form["distance_km"])
            remaining_stops = int(request.form["remaining_stops"])
            previous_delay_min = float(
                request.form["previous_delay_min"]
            )
            weather_severity = float(
                request.form["weather_severity"]
            )
            congestion_index = float(
                request.form["congestion_index"]
            )
            train_type_code = int(
                request.form["train_type_code"]
            )

            if distance_km < 0 or remaining_stops < 0:
                raise ValueError(
                    "Distance and remaining stops cannot be negative."
                )

            # Derive calendar features from scheduled arrival.
            features = {
                "distance_km": distance_km,
                "scheduled_hour": scheduled_arrival.hour,
                "day_of_week": scheduled_arrival.weekday(),
                "month": scheduled_arrival.month,
                "remaining_stops": remaining_stops,
                "previous_delay_min": previous_delay_min,
                "weather_severity": weather_severity,
                "congestion_index": congestion_index,
                "train_type_code": train_type_code,
            }

            input_data = pd.DataFrame(
                [features],
                columns=FEATURE_COLUMNS
            )

            prediction = model.predict(input_data)
            predicted_delay = float(prediction[0])

            if not pd.notna(predicted_delay):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            predicted_delay = round(predicted_delay, 2)

            estimated_arrival = (
                scheduled_arrival
                + timedelta(minutes=predicted_delay)
            )

            if predicted_delay <= 5:
                status = "On Time"
            else:
                status = "Delayed"

            result = {
                "train_number": train_number,
                "scheduled_arrival": scheduled_arrival.strftime(
                    "%d-%m-%Y %I:%M %p"
                ),
                "predicted_delay": predicted_delay,
                "estimated_arrival": estimated_arrival.strftime(
                    "%d-%m-%Y %I:%M %p"
                ),
                "status": status,
            }

        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc) or "Please check the entered values."

        except Exception:
            app.logger.exception("Train ETA prediction failed")
            error = (
                "Prediction failed. Check the model file and input values."
            )

    return render_template(
        "index.html",
        result=result,
        metrics=metrics,
        error=error,
        username=session.get("identifier")
    )


# --------------------------------------------------
# 8. LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# --------------------------------------------------
# 9. HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "running",
        "database": "initialized",
        "model_loaded": model is not None,
        "model_error": model_error,
    }


# --------------------------------------------------
# 10. APPLICATION START
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5001))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
