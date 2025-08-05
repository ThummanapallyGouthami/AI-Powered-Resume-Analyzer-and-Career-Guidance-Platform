# utils/auth.py
USERS = {
    "admin": "1234",
    "user": "abcd"
}

def authenticate(username, password):
    return USERS.get(username) == password
