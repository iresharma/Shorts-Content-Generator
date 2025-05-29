from flask import Flask, request, jsonify
from functools import wraps
import os
import jwt
from jwt.exceptions import InvalidTokenError
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Clerk configuration
CLERK_SECRET_KEY = os.environ.get('CLERK_SECRET_KEY')
CLERK_JWT_KEY = os.environ.get('CLERK_JWT_KEY')  # PEM public key from Clerk


def verify_clerk_token(token):
    """Verify Clerk JWT token"""
    if not token:
        return None

    # Remove Bearer prefix if present
    if token.startswith('Bearer '):
        token = token[7:]

    try:
        # If you have the JWT public key
        if CLERK_JWT_KEY:
            decoded = jwt.decode(
                token,
                CLERK_JWT_KEY,
                algorithms=['RS256'],
                options={"verify_signature": True}
            )
            return decoded

        # Alternative: verify with Clerk API (simpler but requires API call)
        headers = {
            'Authorization': f'Bearer {CLERK_SECRET_KEY}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            'https://api.clerk.com/v1/sessions/verify',
            headers=headers,
            json={'token': token}
        )

        if response.status_code == 200:
            return response.json()
        return None

    except InvalidTokenError:
        return None
    except Exception as e:
        print(f"Token verification error: {e}")
        return None


def require_auth(f):
    """Decorator to require authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401

        user_data = verify_clerk_token(auth_header)

        if not user_data:
            return jsonify({'error': 'Invalid token'}), 401

        # Add user data to request
        request.user = user_data
        return f(*args, **kwargs)

    return decorated_function


# Routes
@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/user')
@require_auth
def get_user():
    """Get current user info"""
    return jsonify({
        'user': request.user,
        'user_id': request.user.get('sub') or request.user.get('user_id')
    })


@app.route('/api/protected-data')
@require_auth
def protected_data():
    """Example protected endpoint"""
    user_id = request.user.get('sub') or request.user.get('user_id')

    # Your business logic here
    data = {
        'message': f'Hello user {user_id}!',
        'timestamp': '2025-05-30',
        'data': ['item1', 'item2', 'item3']
    }

    return jsonify(data)


@app.route('/api/create-post', methods=['POST'])
@require_auth
def create_post():
    """Example POST endpoint"""
    user_id = request.user.get('sub') or request.user.get('user_id')
    data = request.get_json()

    # Your business logic here
    post = {
        'id': 123,
        'user_id': user_id,
        'title': data.get('title'),
        'content': data.get('content'),
        'created_at': '2025-05-30T10:00:00Z'
    }

    return jsonify(post), 201


# CORS setup (if needed)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')  # Your React app URL
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle preflight OPTIONS requests"""
    return '', 200


if __name__ == '__main__':
    if not CLERK_SECRET_KEY:
        print("Warning: CLERK_SECRET_KEY not set")

    app.run(debug=True, host='0.0.0.0', port=5000)