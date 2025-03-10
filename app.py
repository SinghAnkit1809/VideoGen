import streamlit as st
import asyncio
import base64
import nest_asyncio
import os
import time
from pathlib import Path
import tempfile
from video_gen import VideoGenerator

# Apply nest_asyncio to allow asyncio to work with Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Lunrack AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1e2030 100%);
    }
    .stButton > button {
        background-color: #4b61d1;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #6c7fe0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stTextArea > div > div > textarea {
        background-color: #1e2030;
        color: #e0e0e0;
        border: 1px solid #4b61d1;
        border-radius: 8px;
    }
    .css-1d391kg {
        background-color: #1e2030;
    }
    .stSidebar {
        background-color: #161a2b;
    }
    .stSidebar [data-testid="stVerticalBlock"] {
        background-color: #161a2b;
    }
    .stSelectbox label, .stSlider label {
        color: #c0d0f0 !important;
    }
    .css-81oif8 {
        color: #c0d0f0 !important;
    }
    .css-1qg05tj {
        color: #c0d0f0 !important;
    }
    .css-1qg05tj:hover {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #4b61d1;
    }
    .social-icons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .social-icons a {
        display: inline-flex;
        align-items: center;
        padding: 8px 15px;
        background-color: #2a3050;
        border-radius: 50px;
        text-decoration: none;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
        margin: 5px;
    }
    .social-icons a:hover {
        background-color: #4b61d1;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .social-icons img {
        height: 24px;
        margin-right: 8px;
    }
    .stProgress .css-1hrcrf8 {
        background-color: #4b61d1;
    }
    .download-btn {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        border-radius: 8px;
        margin: 10px 0;
        transition: background-color 0.3s;
        font-weight: bold;
    }
    .download-btn:hover {
        background-color: #45a049;
        text-decoration: none;
    }
    .status-info {
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(75, 97, 209, 0.2);
        border-left: 5px solid #4b61d1;
        margin: 15px 0;
    }

    /* Video player container styles */
    .stVideo {
        max-width: 100% !important;
    }
    
    /* Vertical video (9:16) specific styles */
    [data-testid="stVideo"] video {
        max-height: 80vh !important;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    /* Make sure video doesn't overflow container */
    .element-container:has([data-testid="stVideo"]) {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Define voice options
VOICE_OPTIONS = {
    "Christopher (US Male)": "en-US-ChristopherNeural",
    "Jenny (US Female)": "en-US-JennyNeural",
    "Ryan (UK Male)": "en-GB-RyanNeural",
    "Sonia (UK Female)": "en-GB-SoniaNeural",
    "Guy (US Male)": "en-US-GuyNeural"
}

VIDEO_STYLES = {
    "realistic": "Photorealistic, detailed imagery",
    "comics": "Comic book style with bold outlines",
    "anime": "Japanese animation style",
    "line_diagram": "Simple line drawing style with rough yellow background",
    "pixelart": "Retro 8-bit pixel art style from classic games",
    "papercraft": "Paper cutout collage style with layered elements",
    "watercolor": "Soft, artistic watercolor painting",
    "synthwave": "synthwave aesthetic, neon colors, retro-futuristic, 80s style",
    "minimalist": "minimalist style, clean lines, simple shapes, limited color palette",
    "3d_render": "3D rendered scene, computer graphics, detailed textures, volumetric lighting"
}

# Social media icons and links
social_icons = """
<div class="social-icons">
    <a href="https://x.com/A9kitSingh" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path></svg>
        X/Twitter
    </a>
    <a href="https://t.me/SmartLunaBot" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.2 6.4L3.8 12.9c-.5.2-.5.9 0 1.1l5.3 2.9L15 11"></path><path d="M9.1 16.9l1.2 6c.1.5.7.7 1.1.3l3-3"></path><path d="M9.1 16.9L8 9H14L21.1 6.4"></path></svg>
        Telegram
    </a>
    <a href="https://instagram.com/LUNARCKAI" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line></svg>
        Instagram
    </a>
    <a href="https://youtube.com/@BlueBotArtists" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"></path><polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon></svg>
        YouTube
    </a>
</div>
"""

# Utility function to create download link from bytes
def get_download_link_from_bytes(video_bytes, filename="video.mp4", link_text="Download Video"):
    b64 = base64.b64encode(video_bytes.getvalue()).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
    return href

# Display social media handles at top
st.markdown(social_icons, unsafe_allow_html=True)

# Main content area with subtle gradient background
st.markdown("""
<div style="text-align: center; padding: 10px; margin-bottom: 30px;">
    <h1 style="color: #4b61d1; font-size: 2.8em; margin-bottom: 0px;">🎬 Lunarck AI Video Generator</h1>
    <p style="color: #c0d0f0; font-size: 1.2em; max-width: 800px; margin: 10px auto;">Generate engaging AI videos from any topic in seconds!</p>
</div>
""", unsafe_allow_html=True)

# Create sidebar for settings
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0px 20px 0px;">
    <h2 style="color: #4b61d1;">⚙️ Video Settings</h2>
</div>
""", unsafe_allow_html=True)

# Video aspect ratio
aspect_ratio = st.sidebar.selectbox(
    "Aspect Ratio",
    options=["9:16 (Vertical/Mobile)", "16:9 (Horizontal/Landscape)"],
    index=0
)

# Extract the actual ratio
aspect_ratio_value = aspect_ratio.split()[0]

# Video duration
duration = st.sidebar.select_slider(
    "Duration (seconds)",
    options=[30, 45, 60, 90, 120, 180],
    value=45
)

# Video style selection
style_name = st.sidebar.selectbox(
    "Video Style",
    options=list(VIDEO_STYLES.keys()),
    index=0,
    help="Choose the visual style for your video"
)
selected_style = style_name

# Voice selection
voice_name = st.sidebar.selectbox(
    "Narrator Voice",
    options=list(VOICE_OPTIONS.keys()),
    index=0
)
selected_voice = VOICE_OPTIONS[voice_name]

# Explanation in sidebar
st.sidebar.markdown("""
<div style="background-color: #1e2030; padding: 20px; border-radius: 10px; margin-top: 30px;">
    <h3 style="color: #4b61d1; text-align: center;">How it works</h3>
    <ol style="color: #c0d0f0;">
        <li>Enter a topic for your video</li>
        <li>Our AI will craft a story around your topic</li>
        <li>Generate images for each story segment</li>
        <li>Add narration with your selected voice</li>
        <li>Download the final video</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Create a container with better styling
main_container = st.container()
with main_container:
    # Input for video topic with better styling
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_area("What should your video be about?", 
                            height=100,
                            placeholder="Enter a topic, theme, or story idea... (e.g. 'A superhero who discovers powers during a thunderstorm')",
                            help="Be specific to get better results. Include characters, settings, or events you want in your story.")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        generate_button = st.button("🚀 Generate Video", disabled=not topic.strip(), use_container_width=True)

# Progress updates
progress_placeholder = st.empty()
video_placeholder = st.empty()

# Initialize session state
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'generating' not in st.session_state:
    st.session_state.generating = False
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

async def update_progress(message):
    """Update progress message in the Streamlit app"""
    progress_placeholder.markdown(f"""
    <div class="status-info">
        <h3 style="margin-bottom: 5px;">{message}</h3>
        <div class="progress-bar"></div>
    </div>
    """, unsafe_allow_html=True)

async def generate_video():
    """Handle video generation process"""
    try:
        st.session_state.generating = True
        
        # Create VideoGenerator with selected settings
        generator = VideoGenerator(
            aspect_ratio=aspect_ratio_value,
            duration=duration,
            voice=selected_voice,
            style=selected_style  # Add this line
        )
        
        # Generate video and get bytes
        video_bytes, story = await generator.generate_video(topic, update_progress)
        
        # Save bytes to a temporary file for Streamlit's video player
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_bytes.getvalue())
        temp_file.close()
        
        # Store results in session state
        st.session_state.video_bytes = video_bytes
        st.session_state.temp_file_path = temp_file.name
        st.session_state.story = story
        st.session_state.generating = False
        
        # Force refresh
        st.rerun()
        
    except Exception as e:
        st.session_state.generating = False
        progress_placeholder.markdown(f"""
        <div style="padding: 15px; border-radius: 8px; background-color: rgba(255, 70, 70, 0.2); border-left: 5px solid #ff4646; margin: 15px 0;">
            <h3 style="color: #ff4646; margin-bottom: 5px;">Error</h3>
            <p>{str(e)}</p>
        </div>
        """, unsafe_allow_html=True)

# Run the async function if button clicked
if generate_button:
    # Run the async function
    asyncio.run(generate_video())

# Show generating status
if st.session_state.generating:
    progress_placeholder.markdown("""
    <div class="status-info">
        <h3 style="margin-bottom: 5px;">🔄 Generating your video...</h3>
        <p>This may take a few minutes. We're creating something special for you!</p>
    </div>
    """, unsafe_allow_html=True)
    st.spinner("Working on your masterpiece...")

# Display video if available
if st.session_state.video_bytes and st.session_state.temp_file_path:
    # Display in an attractive container
    st.markdown("""
    <div style="background-color: #1e2030; padding: 20px; border-radius: 15px; margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h2 style="color: #4b61d1; text-align: center; margin-bottom: 20px;">✨ Your Video is Ready!</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display video in a more attractive way
    video_placeholder.video(st.session_state.temp_file_path)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Download button with better styling
        st.markdown(get_download_link_from_bytes(
            st.session_state.video_bytes, 
            filename=f"lunarck_video_{int(time.time())}.mp4", 
            link_text="⬇️ Download Video"
        ), unsafe_allow_html=True)
    
    with col2:
        # Regenerate option
        if st.button("🔄 Generate Another Video", use_container_width=True):
            # Clean up temp file
            if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                try:
                    os.unlink(st.session_state.temp_file_path)
                except:
                    pass
            
            st.session_state.video_bytes = None
            st.session_state.temp_file_path = None
            st.rerun()
    
    # Display the generated story if available
    if hasattr(st.session_state, 'story'):
        with st.expander("📖 View Generated Story"):
            st.markdown(f"""
            <div style="background-color: #242838; padding: 20px; border-radius: 10px; margin-top: 10px;">
                <p style="color: #e0e0e0; line-height: 1.6;">{st.session_state.story}</p>
            </div>
            """, unsafe_allow_html=True)

# Clean up on app exit or refresh
def cleanup():
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.unlink(st.session_state.temp_file_path)
        except:
            pass

# Register cleanup handler
import atexit
atexit.register(cleanup)

# Footer
st.markdown("""
<div style="background-color: #161a2b; padding: 20px; border-radius: 10px; margin-top: 50px; text-align: center;">
    <p style="color: #c0d0f0; margin-bottom: 10px;">Made with ❤️ by Lunarck AI Video Generator</p>
    <div class="social-icons" style="justify-content: center; margin-top: 15px;">
        <a href="https://x.com/A9kitSingh" target="_blank" style="font-size: 0.9em; padding: 5px 10px;">
            X/Twitter
        </a>
        <a href="https://t.me/SmartLunaBot" target="_blank" style="font-size: 0.9em; padding: 5px 10px;">
            Telegram
        </a>
        <a href="https://instagram.com/LUNARCKAI" target="_blank" style="font-size: 0.9em; padding: 5px 10px;">
            Instagram
        </a>
        <a href="https://youtube.com/@BlueBotArtists" target="_blank" style="font-size: 0.9em; padding: 5px 10px;">
            YouTube
        </a>
    </div>
</div>
""", unsafe_allow_html=True)