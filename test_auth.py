import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="Auth Test",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple configuration
config = {
    'credentials': {
        'usernames': {
            'testuser': {
                'email': 'test@example.com',
                'name': 'Test User',
                'password': stauth.Hasher(['test123']).generate()[0]  # Hashed password
            }
        }
    },
    'cookie': {
        'expiry_days': 1,
        'key': 'test_key_123',
        'name': 'test_cookie_name'
    },
    'preauthorized': {
        'emails': []
    }
}

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Custom CSS for better visibility
st.markdown("""
<style>
    .stApp {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 2px solid #dfe1e5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    st.title("🔒 Authentication Test")
    
    # Login form
    name, authentication_status, username = authenticator.login('Login', 'main')
    
    # Handle authentication status
    if authentication_status:
        st.success(f'Welcome *{name}*')
        st.balloons()
        if authenticator.logout('Logout', 'main'):
            st.experimental_rerun()
        
        # Show some protected content
        st.title('🎉 Successfully Logged In!')
        st.write("You now have access to the protected content.")
        
    elif authentication_status is False:
        st.error('❌ Username/password is incorrect')
    else:
        st.warning('ℹ️ Please enter your username and password')
        
        # Show test credentials
        st.markdown("---")
        st.markdown("### Test Credentials")
        st.code("""
        Username: testuser
        Password: test123
        """)

if __name__ == "__main__":
    main()
