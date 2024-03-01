from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# In-memory user database (replace this with a database of your choice in a production environment)
users_db = {}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users_db:
            return 'Username already exists, choose another one.'
        
        users_db[username] = {'password': password}

        return 'Signup successful. You can now login.'

    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    if username in users_db and users_db[username]['password'] == password:
        return f'Login successful, Welcome {username}!'
    else:
        return 'Login failed. Check your username and password.'

if __name__ == '__main__':
    app.run(port=5001, debug=True)
