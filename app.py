
"""
RailPulse AI
AI-Based Dynamic Train ETA Prediction and Delay Forecasting
Using XGBoost Algorithm
"""

from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps
import sqlite3
import json

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

DB_PATH = ROOT / "users.db"
MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = Flask(__name__)

app.secret_key = "RailPulse_AI_Secret_Key_2026"


# ==================================================
# 2. MODEL CONFIGURATION
# ==================================================

FEATURE_COLUMNS = [
    "distance_km",
    "scheduled_hour",
    "day_of_week",
    "month",
    "remaining_stops",
    "previous_delay_min",
    "weather_severity",
    "congestion_index",
    "train_type_code"
]

model = None
metrics = {}


# ==================================================
# 3. DATABASE INITIALIZATION
# ==================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create users table if it does not exist."""

    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password TEXT
        )
    """)

    connection.commit()
    connection.close()


def update_database():
    """Ensure the existing database has the required columns."""

    connection = get_connection()

    columns = [
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(users)"
        ).fetchall()
    ]

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
    connection.close()


# ==================================================
# 4. VALIDATION FUNCTIONS
# ==================================================

def valid_identifier(identifier):
    """Check email address or phone number."""

    identifier = identifier.strip()

    if "@" in identifier:
        parts = identifier.split("@")

        return (
            len(parts) == 2
            and bool(parts[0])
            and "." in parts[1]
            and bool(parts[1].split(".")[0])
        )

    phone = (
        identifier
        .replace("+", "")
        .replace("-", "")
        .replace(" ", "")
    )

    return phone.isdigit() and 10 <= len(phone) <= 15


def valid_password(password):
    """Require at least seven alphanumeric characters."""

    return len(password) >= 7 and password.isalnum()


# ==================================================
# 5. LOGIN PROTECTION
# ==================================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


# ==================================================
# 6. REGISTRATION
# ==================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        identifier = request.form.get(
            "identifier", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not valid_identifier(identifier):
            return render_template(
                "auth.html",
                mode="register",
                error="Enter a valid email address or phone number."
            )

        if not valid_password(password):
            return render_template(
                "auth.html",
                mode="register",
                error="Password must contain at least 7 letters or digits."
            )

        if password != confirm_password:
            return render_template(
                "auth.html",
                mode="register",
                error="Passwords do not match."
            )

        hashed_password = generate_password_hash(password)

        connection = get_connection()

        try:

            columns = [
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(users)"
                ).fetchall()
            ]

            # Save to both password columns when available.
            # password_hash is the main authentication column.

            if "password_hash" in columns and "password" in columns:

                connection.execute(
                    """
                    INSERT INTO users
                    (identifier, password_hash, password)
                    VALUES (?, ?, ?)
                    """,
                    (
                        identifier,
                        hashed_password,
                        hashed_password
                    )
                )

            elif "password_hash" in columns:

                connection.execute(
                    """
                    INSERT INTO users
                    (identifier, password_hash)
                    VALUES (?, ?)
                    """,
                    (identifier, hashed_password)
                )

            else:

                connection.execute(
                    """
                    INSERT INTO users
                    (identifier, password)
                    VALUES (?, ?)
                    """,
                    (identifier, hashed_password)
                )

            connection.commit()

            user = connection.execute(
                """
                SELECT id, identifier
                FROM users
                WHERE identifier = ?
                """,
                (identifier,)
            ).fetchone()

            if user is None:
                raise sqlite3.Error("Account could not be retrieved.")

            # Create login session after successful registration.

            session.clear()
            session["user_id"] = user["id"]
            session["identifier"] = user["identifier"]

            return redirect(url_for("home"))

        except sqlite3.IntegrityError:

            return render_template(
                "auth.html",
                mode="register",
                error="This email or phone number is already registered."
            )

        except sqlite3.Error:

            app.logger.exception("Registration database error")

            return render_template(
                "auth.html",
                mode="register",
                error="Database error. Please try again."
            )

        finally:
            connection.close()

    return render_template(
        "auth.html",
        mode="register"
    )


# ==================================================
# 7. LOGIN
# ==================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        identifier = request.form.get(
            "identifier", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        connection = get_connection()

        try:

            user = connection.execute(
                """
                SELECT *
                FROM users
                WHERE identifier = ?
                """,
                (identifier,)
            ).fetchone()

            stored_password = None

            if user:

                # Prefer password_hash, then use password
                # for compatibility with older database records.

                if (
                    "password_hash" in user.keys()
                    and user["password_hash"]
                ):
                    stored_password = user["password_hash"]

                elif (
                    "password" in user.keys()
                    and user["password"]
                ):
                    stored_password = user["password"]

        except sqlite3.Error:

            app.logger.exception("Login database error")

            return render_template(
                "auth.html",
                mode="login",
                error="Database error. Please try again."
            )

        finally:
            connection.close()

        if user and stored_password:

            try:
                password_matches = check_password_hash(
                    stored_password,
                    password
                )
            except (ValueError, TypeError):
                password_matches = False

            if password_matches:

                session.clear()

                session["user_id"] = user["id"]
                session["identifier"] = user["identifier"]

                session.permanent = True

                return redirect(url_for("home"))

        return render_template(
            "auth.html",
            mode="login",
            error="Invalid email, phone number or password."
        )

    return render_template(
        "auth.html",
        mode="login"
    )


# ==================================================
# 8. LOGOUT
# ==================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==================================================
# 9. LOAD XGBOOST MODEL
# ==================================================

def load_model():

    global model, metrics

    if not MODEL_PATH.exists():

        app.logger.error(
            "Model file not found: %s",
            MODEL_PATH
        )

        model = None
        return

    try:

        model = joblib.load(MODEL_PATH)

        app.logger.info(
            "XGBoost model loaded successfully."
        )

    except Exception:

        app.logger.exception(
            "Error loading XGBoost model."
        )

        model = None

    if METRICS_PATH.exists():

        try:

            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8"
            ) as file:

                metrics = json.load(file)

        except (OSError, json.JSONDecodeError):

            app.logger.exception(
                "Could not load model metrics."
            )

            metrics = {}


# ==================================================
# 10. TRAIN ETA DASHBOARD
# ==================================================

@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    result = None
    error = None

    if request.method == "POST":

        if model is None:

            error = (
                "XGBoost model is not loaded. "
                "Check the models folder."
            )

        else:

            try:

                train_number = request.form.get(
                    "train_number", ""
                ).strip()

                scheduled_text = request.form.get(
                    "scheduled_arrival", ""
                )

                distance_km = float(
                    request.form.get("distance_km", 0)
                )

                remaining_stops = int(
                    request.form.get("remaining_stops", 0)
                )

                previous_delay_min = float(
                    request.form.get("previous_delay_min", 0)
                )

                weather_severity = float(
                    request.form.get("weather_severity", 0)
                )

                congestion_index = float(
                    request.form.get("congestion_index", 0)
                )

                train_type_code = int(
                    request.form.get("train_type_code", 0)
                )

                if not train_number:
                    raise ValueError("Enter the train number.")

                if distance_km <= 0:
                    raise ValueError(
                        "Distance must be greater than zero."
                    )

                if remaining_stops < 0:
                    raise ValueError(
                        "Remaining stops cannot be negative."
                    )

                if previous_delay_min < 0:
                    raise ValueError(
                        "Previous delay cannot be negative."
                    )

                if not 0 <= weather_severity <= 5:
                    raise ValueError(
                        "Weather severity must be between 0 and 5."
                    )

                if not 0 <= congestion_index <= 1:
                    raise ValueError(
                        "Congestion index must be between 0 and 1."
                    )

                if train_type_code < 0:
                    raise ValueError(
                        "Train type code cannot be negative."
                    )

                scheduled_arrival = datetime.strptime(
                    scheduled_text,
                    "%Y-%m-%dT%H:%M"
                )

                input_data = pd.DataFrame(
                    [{
                        "distance_km": distance_km,
                        "scheduled_hour": scheduled_arrival.hour,
                        "day_of_week": scheduled_arrival.weekday(),
                        "month": scheduled_arrival.month,
                        "remaining_stops": remaining_stops,
                        "previous_delay_min": previous_delay_min,
                        "weather_severity": weather_severity,
                        "congestion_index": congestion_index,
                        "train_type_code": train_type_code
                    }],
                    columns=FEATURE_COLUMNS
                )

                predicted_delay = float(
                    model.predict(input_data)[0]
                )

                predicted_delay = max(
                    0,
                    predicted_delay
                )

                estimated_arrival = (
                    scheduled_arrival
                    + timedelta(minutes=predicted_delay)
                )

                if predicted_delay <= 5:
                    status = "On Time"

                elif predicted_delay <= 30:
                    status = "Slightly Delayed"

                else:
                    status = "Highly Delayed"

                result = {
                    "train_number": train_number,
                    "scheduled_arrival": (
                        scheduled_arrival.strftime(
                            "%d-%m-%Y %I:%M %p"
                        )
                    ),
                    "predicted_delay": round(
                        predicted_delay, 2
                    ),
                    "estimated_arrival": (
                        estimated_arrival.strftime(
                            "%d-%m-%Y %I:%M %p"
                        )
                    ),
                    "status": status
                }

            except ValueError as exception:

                error = str(exception)

            except Exception:

                app.logger.exception(
                    "Train ETA prediction error."
                )

                error = (
                    "Prediction failed. "
                    "Please check your inputs and model."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
        metrics=metrics,
        username=session.get("identifier"),
        current_year=datetime.now().year
    )


# ==================================================
# 11. APPLICATION STARTUP
# ==================================================

if __name__ == "__main__":

    init_db()
    update_database()
    load_model()

    print("\n====================================")
    print("         RAILPULSE AI")
    print("====================================")
    print("Login:     http://127.0.0.1:5001/login")
    print("Register:  http://127.0.0.1:5001/register")
    print("Dashboard: http://127.0.0.1:5001/")
    print("====================================\n")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )
