import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import os
from pathlib import Path

# Path to the users.yaml file
USERS_FILE = Path(__file__).parent / ".streamlit" / "users.yaml"

# Create .streamlit directory if it doesn't exist
USERS_FILE.parent.mkdir(exist_ok=True)

def load_config():
    """Load or create the authentication configuration."""
    if not USERS_FILE.exists():
        # Create a default config if it doesn't exist
        config = {
            'credentials': {
                'usernames': {}
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'some_signature_key',  # Should be random and secret in production
                'name': 'spars_auth_cookie'
            },
            'preauthorized': {
                'emails': []
            }
        }
        save_config(config)
        return config
    
    with open(USERS_FILE, 'r') as file:
        return yaml.load(file, Loader=SafeLoader)

def save_config(config):
    """Save the authentication configuration to the YAML file."""
    with open(USERS_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def register_user(username: str, name: str, password: str):
    """Register a new user."""
    config = load_config()
    
    if username in config['credentials']['usernames']:
        raise ValueError('Username already exists')
    
    # Hash the password
    hashed_password = stauth.Hasher([password]).generate()[0]
    
    # Add the new user
    config['credentials']['usernames'][username] = {
        'email': f"{username}@example.com",  # You might want to collect real emails
        'name': name,
        'password': hashed_password
    }
    
    save_config(config)
    return True

def get_authenticator():
    """Get an instance of the authenticator."""
    config = load_config()
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    
    return authenticator

def get_authenticator():
    """Initialize and return the authenticator with default test user if no users exist."""
    config_file = Path(__file__).parent / ".streamlit" / "users.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Default config with test user
    default_config = {
        'credentials': {
            'usernames': {
                'testuser': {
                    'email': 'test@example.com',
                    'name': 'Test User',
                    'password': stauth.Hasher(['test123']).generate()[0]
                }
            }
        }
    }
    
    # Create config file with default user if it doesn't exist
    if not config_file.exists():
        with open(config_file, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
    
    # Load existing config
    with open(config_file) as file:
        config = yaml.load(file, Loader=SafeLoader) or default_config
    
    # Ensure the config has the expected structure
    if 'credentials' not in config or 'usernames' not in config['credentials']:
        config = default_config
    
    return stauth.Authenticate(
        config['credentials'],
        'auth_cookie_name',
        'auth_signature_key',
        cookie_expiry_days=30
    )

def register_user(username, name, password):
    """Register a new user."""
    config_file = Path(__file__).parent / ".streamlit" / "users.yaml"
    authenticator = get_authenticator()
    
    try:
        # Check if username already exists
        if username in authenticator.credentials['usernames']:
            raise ValueError("Username already exists")
            
        # Register the new user
        hashed_password = stauth.Hasher([password]).generate()[0]
        
        # Update config
        with open(config_file) as file:
            config = yaml.load(file, Loader=SafeLoader)
        
        if config is None:
            config = {'credentials': {'usernames': {}}}
        
        config['credentials']['usernames'][username] = {
            'email': f"{username}@example.com",
            'name': name,
            'password': hashed_password
        }
        
        with open(config_file, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
    except Exception as e:
        raise ValueError(f"Registration failed: {str(e)}")

def login(authenticator):
    """Display login/signup form and handle authentication.
    
    Args:
        authenticator: The authenticator instance to use for login.
        
    Returns:
        tuple: (authentication_status, username)
    """
    st.markdown("""
    <style>
        .auth-container {
            max-width: 500px;
            margin: 2rem auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            width: 50%;
            padding: 0;
            color: #4a4a4a;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
            color: #1e88e5;
            border-bottom: 3px solid #1e88e5;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f0f2f6;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        .stTextInput>div>div>input {
            padding: 0.75rem 1rem;
            border-radius: 8px;
            border: 2px solid #dfe1e5;
        }
        .stTextInput>div>div>input:focus {
            border-color: #1e88e5;
            box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
        }
        .error-message {
            color: #d32f2f;
            font-weight: 500;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #ffebee;
            border-radius: 4px;
            border-left: 4px solid #d32f2f;
        }
        .success-message {
            color: #2e7d32;
            font-weight: 500;
            margin: 0.5rem 0;
            padding: 0.5rem;
            background-color: #e8f5e9;
            border-radius: 4px;
            border-left: 4px solid #2e7d32;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a container for the login/signup form
    with st.container():
        st.title("🔒 Welcome to SPARS")
        st.markdown("---")

        # Create tabs for login/signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            st.header("Login to Your Account")
            
            # Display login form using the authenticator
            name, authentication_status, username = authenticator.login('Login', 'main')
            
            # Handle authentication status
            if authentication_status is False:
                st.error("❌ Invalid username or password")
            elif authentication_status is None:
                # Show default credentials hint
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <p>Try these default credentials:</p>
                    <p>👤 Username: <code>testuser</code></p>
                    <p>🔑 Password: <code>test123</code></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                return True, username

        with tab2:
            st.header("Create a New Account")
            with st.form("signup_form"):
                new_username = st.text_input("👤 Username", placeholder="Choose a username")
                new_name = st.text_input("📝 Full Name", placeholder="Your full name")
                new_password = st.text_input("🔑 Password", type="password", placeholder="Create a strong password")
                confirm_password = st.text_input("✅ Confirm Password", type="password", placeholder="Retype your password")

                submitted = st.form_submit_button("Create Account")
                if submitted:
                    if not all([new_username, new_name, new_password, confirm_password]):
                        st.error("⚠️ Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("⚠️ Passwords do not match!")
                    elif len(new_password) < 6:
                        st.error("⚠️ Password must be at least 6 characters long")
                    else:
                        try:
                            register_user(new_username, new_name, new_password)
                            st.success("✅ Account created successfully! Please log in.")
                        except ValueError as e:
                            st.error(f"⚠️ {str(e)}")

    return False, None

def check_auth():
    """Check if user is authenticated, redirect to login if not."""
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    
    if st.session_state.get('authentication_status') != True:
        st.session_state.authentication_status, st.session_state.username = login()
        st.stop()
    
    return st.session_state.username
