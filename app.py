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
    page_title="AI Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define voice options
VOICE_OPTIONS = {
    "Christopher (US Male)": "en-US-ChristopherNeural",
    "Jenny (US Female)": "en-US-JennyNeural",
    "Ryan (UK Male)": "en-GB-RyanNeural",
    "Sonia (UK Female)": "en-GB-SoniaNeural",
    "Guy (US Male)": "en-US-GuyNeural"
}

# Utility function to create download link from bytes
def get_download_link_from_bytes(video_bytes, filename="video.mp4", link_text="Download Video"):
    b64 = base64.b64encode(video_bytes.getvalue()).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Create sidebar for settings
st.sidebar.title("Video Settings")

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

# Voice selection
voice_name = st.sidebar.selectbox(
    "Narrator Voice",
    options=list(VOICE_OPTIONS.keys()),
    index=0
)
selected_voice = VOICE_OPTIONS[voice_name]

# Explanation in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### How it works
1. Enter a topic for your video
2. Our AI will craft a story around your topic
3. Generate images for each story segment
4. Add narration with your selected voice
5. Download the final video
""")

# Main content area
st.title("🎬 Lunarck AI Video Generator")
st.markdown("Generate engaging AI videos from any topic in seconds!")

# Input for video topic
topic = st.text_area("What should your video be about?", 
                     height=100,
                     placeholder="Enter a topic, theme, or story idea... (e.g. 'A superhero who discovers powers during a thunderstorm')",
                     help="Be specific to get better results. Include characters, settings, or events you want in your story.")

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
    progress_placeholder.markdown(f"### {message}")

async def generate_video():
    """Handle video generation process"""
    try:
        st.session_state.generating = True
        
        # Create VideoGenerator with selected settings
        generator = VideoGenerator(
            aspect_ratio=aspect_ratio_value,
            duration=duration,
            voice=selected_voice
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
        st.session_state.generating = False
        
        # Force refresh
        st.rerun()
        
    except Exception as e:
        st.session_state.generating = False
        progress_placeholder.error(f"Error: {str(e)}")

# Generate button
if st.button("Generate Video", disabled=st.session_state.generating or not topic.strip()):
    # Run the async function
    asyncio.run(generate_video())

# Show generating status
if st.session_state.generating:
    progress_placeholder.info("Generating your video... This may take a few minutes.")
    st.spinner("Working on your masterpiece...")

# Display video if available
if st.session_state.video_bytes and st.session_state.temp_file_path:
    # Show video from temp file
    video_placeholder.markdown("### Your Video")
    video_placeholder.video(st.session_state.temp_file_path)
    
    # Download button
    st.markdown(get_download_link_from_bytes(
        st.session_state.video_bytes, 
        filename=f"video_{int(time.time())}.mp4", 
        link_text="⬇️ Download Video"
    ), unsafe_allow_html=True)
    
    # Regenerate option
    if st.button("Generate Another Video"):
        # Clean up temp file
        if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
            try:
                os.unlink(st.session_state.temp_file_path)
            except:
                pass
        
        st.session_state.video_bytes = None
        st.session_state.temp_file_path = None
        st.rerun()

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
st.markdown("---")
st.markdown("Made with ❤️ by Lunarck AI Video Generator")