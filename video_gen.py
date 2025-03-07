import os
import math
import time
import re
import json
import random
import requests
import asyncio
import edge_tts
import sys
from pathlib import Path
from io import BytesIO
import numpy as np
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
from tempfile import TemporaryDirectory, gettempdir

class VideoGenerator:
    def __init__(self, aspect_ratio="9:16", duration=45, voice=None):
        self.model = "openai"
        # Set dimensions based on aspect ratio
        if aspect_ratio == "9:16":
            self.width = 720
            self.height = 1280
            self.is_vertical = True
        else:  # 16:9
            self.width = 1280
            self.height = 720
            self.is_vertical = False
            
        self.target_duration = duration
        self.max_segment_duration = 4
        self.num_segments = math.ceil(self.target_duration / self.max_segment_duration)
        self.speech_rate = "-10%"
        self.voice = voice  # Store selected voice
        
        # No longer creating output directory since we're not persisting videos

    def download_image(self, prompt, seed, max_retries=3):
        """Download image from pollinations.ai with proper error handling"""
        encoded_prompt = requests.utils.quote(prompt)
        image_url = f"https://pollinations.ai/p/{encoded_prompt}?width={self.width}&height={self.height}&seed={seed}&model=flux&nologo=true"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to download image after {max_retries} attempts: {str(e)}")
                time.sleep(2)
                continue

    async def generate_video(self, topic, progress_callback=None):
        """Main video generation workflow without persistent storage"""
        if hasattr(self, 'last_video'):
            del self.last_video
        import gc; gc.collect()
        
        with TemporaryDirectory(dir=gettempdir()) as temp_dir:  # Force temp dir to /tmp
            temp_path = Path(temp_dir)
            try:
                # Story generation
                if progress_callback:
                    await progress_callback("📝 Crafting your story...")
                story, segments = self._generate_story(topic)
                
                # Image generation
                if progress_callback:
                    await progress_callback("🎨 Creating visuals...")
                image_prompts = self._create_image_prompts(story, segments)
                images = []
                
                for i, prompt in enumerate(image_prompts):
                    for attempt in range(3):
                        try:
                            seed = random.randint(1, 999999)
                            image = self.download_image(prompt, seed)
                            images.append(np.array(image))
                            del image  # Free PIL image memory immediately
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise RuntimeError(f"Failed to generate image {i+1}: {str(e)}")
                            await asyncio.sleep(1)
                
                # Audio generation
                if progress_callback:
                    await progress_callback("🔊 Recording narration...")
                audio_path = await self._generate_audio(segments, temp_path)
                
                # Video compilation
                if progress_callback:
                    await progress_callback("🎥 Compiling final video...")
                video_bytes = await self._compile_video(images, segments, audio_path, temp_path)
                
                # Return the video bytes directly instead of saving to file
                return video_bytes, story
                
            except Exception as e:
                raise RuntimeError(f"Video generation failed: {str(e)}")
            finally:
                # Force clean up temporary files
                try:
                    for f in temp_path.glob("*"):
                        f.unlink()
                except:
                    pass

    # The rest of the methods remain the same
    def _generate_story(self, topic):
        """Generate story segments with proper pacing"""
        prompt = f"""Write a simple, engaging story about {topic} that can be narrated in exactly {self.target_duration} seconds. Follow these rules:
        1. Use simple English (A2/B1 level) with short sentences
        2. Include natural speaking pauses between ideas
        3. Use concrete words over abstract concepts
        4. Limit complex vocabulary (e.g. use 'make' instead of 'fabricate')
        5. Structure with clear cause-effect relationships
        6. Use everyday examples readers can relate to
        seed = {random.randint(0,1000000)}
        
        Example good sentence: "When the rain didn't stop, Mia knew she had to move her garden to higher ground."
        Example bad sentence: "The precipitation persisting, Mia was compelled to relocate her horticultural project elevationally."
        
        Make exactly {self.num_segments} natural segments with oral storytelling flow. Return ONLY the story."""
        
        response = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": "Expert storyteller"},
                    {"role": "user", "content": prompt}
                ],
                "model": self.model,
                "stream": False,
                "seed": random.randint(0, 1000000)
            },
            timeout=60
        )
        response.raise_for_status()
        return self._split_text(response.text.strip())

    def _split_text(self, text):
        """Split text into well-paced segments"""
        # Clean and normalize text first
        text = text.strip()
        if not text:
            return [], ""
            
        segments = []
        story = text
        target_length = len(text) // self.num_segments
        
        # Split by sentences first
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + "."  # Add back the period
            if current_length + len(sentence) <= target_length or not current_segment:
                current_segment.append(sentence)
                current_length += len(sentence)
            else:
                segments.append(" ".join(current_segment))
                current_segment = [sentence]
                current_length = len(sentence)
        
        # Add any remaining segment
        if current_segment:
            segments.append(" ".join(current_segment))
        
        # If we need more segments, split the longest ones
        while len(segments) < self.num_segments and any(len(s) > target_length for s in segments):
            # Find longest segment
            longest_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
            long_segment = segments[longest_idx]
            
            # Split at the nearest sentence or space
            split_point = long_segment.rfind('.', 0, len(long_segment)//2)
            if split_point == -1:
                split_point = long_segment.rfind(' ', 0, len(long_segment)//2)
            
            if split_point != -1:
                segments[longest_idx:longest_idx+1] = [
                    long_segment[:split_point].strip(),
                    long_segment[split_point:].strip()
                ]
        
        # Ensure we have exactly the number of segments needed
        while len(segments) > self.num_segments:
            # Merge shortest adjacent segments
            lengths = [len(s) for s in segments]
            min_combined_idx = min(range(len(segments)-1),
                                 key=lambda i: lengths[i] + lengths[i+1])
            segments[min_combined_idx:min_combined_idx+2] = [
                segments[min_combined_idx] + " " + segments[min_combined_idx+1]
            ]
        
        while len(segments) < self.num_segments:
            # Split longest segment
            max_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
            segment = segments[max_idx]
            split_point = len(segment) // 2
            split_point = segment.rfind(' ', 0, split_point)
            if split_point == -1:
                split_point = len(segment) // 2
            
            segments[max_idx:max_idx+1] = [
                segment[:split_point].strip(),
                segment[split_point:].strip()
            ]
        
        return story, segments

    def _create_image_prompts(self, story, segments):
        """Generate consistent visual prompts"""
        prompt = f"""COMPREHENSIVE VISUAL STORYTELLING GUIDANCE:
    
        1. STORY ANALYSIS:
        Read this story carefully: 
        {story}
    
        2. CHARACTER IDENTIFICATION:
        - Identify ALL people mentioned in the story
        - For each person, list key visual attributes:
            * Approximate age
            * Distinctive physical features
            * Typical clothing/accessories
            * Associated objects/tools/environment
    
        3. SETTING & TIMELINE:
        - Identify ALL locations in the story
        - Note any time period indicators
        - Track visual progression through story
    
        4. FOR EACH SEGMENT CREATE A PROMPT:
        Analyze these story segments:
        {json.dumps(segments)}
        
        PROMPT CREATION RULES:
        - MAINTAIN CHARACTER CONSISTENCY: Same person = identical visual features across segments
        - USE DESCRIPTIVE TERMS: "Middle-aged entrepreneur with glasses" not "John" or "he"
        - INCLUDE SETTING DETAILS: Time period, location, weather if relevant
        - EMOTIONAL TONE: Match visual mood to story segment
        - SPECIFY ACTION: What is happening in this exact moment
        - ART STYLE: Consistent digital art style, cartoonish but detailed
        - MAX 120 CHARACTERS per prompt
        
        Example prompts:
        1. "Young Marie Curie in 1890s Paris lab, examining glowing test tubes, determined expression, cartoon digital art"
        2. "Same scientist measuring radiation with simple tools, dim laboratory, focused expression, cartoon digital art"
        
        Return EXACTLY {len(segments)} prompts in a JSON array with key "prompts". Each prompt must be a single string.
        """
        
        response = requests.post(
            "https://text.pollinations.ai/",
            json={
                "messages": [
                    {"role": "system", "content": "You are a visual consistency expert who creates cohesive image prompts that maintain character continuity and visual storytelling."},
                    {"role": "user", "content": prompt}
                ],
                "model": self.model,
                "jsonMode": True,
                "seed": random.randint(0, 1000000)
            },
            timeout=60
        )
        
        try:
            result = json.loads(response.text)
            if "prompts" in result:
                return [p[:120] for p in result["prompts"]]
            else:
                # Try to extract prompts directly
                if isinstance(result, list):
                    return [p[:120] for p in result]
                else:
                    return [p[:120] for p in list(result.values())]
        except:
            # Fallback option with basic prompts
            return [f"{seg[:80].replace('.', ',')} cartoon digital art style, inspirational scene" for seg in segments]

    async def _generate_audio(self, segments, temp_path):
        # List of good voices to choose from
        voice_options = [
            "en-US-ChristopherNeural",  # Male voice
            "en-US-JennyNeural",        # Female voice
            "en-GB-RyanNeural",         # British male voice
            "en-GB-SoniaNeural",        # British female voice
            "en-US-GuyNeural"           # Another male voice
        ]
        
        # Use selected voice or randomly choose one for consistency
        selected_voice = self.voice if self.voice else random.choice(voice_options)
        
        """Generate and combine audio segments"""
        audio_files = []
        segment_durations = []
        
        for i, text in enumerate(segments):
            path = temp_path / f"audio_{i}.mp3"
            communicate = edge_tts.Communicate(text, selected_voice)
            await communicate.save(str(path))
            audio_files.append(str(path))
            
            # Get duration of segment
            audio = AudioFileClip(str(path))
            segment_durations.append(audio.duration)
            audio.close()
        
        # Combine audio files
        combined = concatenate_audioclips([AudioFileClip(f) for f in audio_files])
        final_path = temp_path / "final_audio.mp3"
        combined.write_audiofile(str(final_path))
        combined.close()
        
        return final_path, segment_durations

    async def _compile_video(self, images, segments, audio_path, temp_path):
        """Create final video with improved temporary file handling"""
        # Force MoviePy to use /tmp directory
        os.environ["FFMPEG_TEMP_DIR"] = str(temp_path)
        
        try:
            audio_path, segment_durations = audio_path
            clips = []
            
            for image_array, segment, duration in zip(images, segments, segment_durations):
                # Calculate words per second for this segment
                words = segment.split()
                words_per_second = len(words) / duration
                
                # Split text into chunks that match speaking rhythm
                chunk_size = max(1, int(words_per_second * 1.5))
                chunks = []
                current_chunk = []
                
                for word in words:
                    current_chunk.append(word)
                    if len(current_chunk) >= chunk_size or word[-1] in '.!?':
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Create sub-clips for each text chunk
                chunk_duration = duration / len(chunks)
                for i, chunk in enumerate(chunks):
                    image_with_caption = self._add_captions(image_array, chunk)
                    sub_clip = ImageClip(image_with_caption).set_duration(chunk_duration)
                    clips.append(sub_clip)
            
            # Combine all clips
            final_clip = concatenate_videoclips(clips)
            final_clip = final_clip.set_audio(AudioFileClip(str(audio_path)))
            
            # Create video file directly in temp_path
            output_path = temp_path / "output.mp4"
            
            # Ensure temp directory exists and is writable
            temp_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(temp_path), 0o777)
            
            # Create temp audio file path
            temp_audio = temp_path / "temp_audio.mp3"
            
            final_clip.write_videofile(
                str(output_path),
                fps=15,
                threads=2,
                preset='ultrafast',
                temp_audiofile=str(temp_audio),
                ffmpeg_params=[
                    '-movflags', '+faststart',
                    '-vf', f'scale={self.width}:{self.height}',
                    '-c:v', 'libx264',
                    '-crf', '28',
                    '-tune', 'fastdecode'
                ],
                logger=None
            )
            
            # Read file into memory
            video_bytes = BytesIO()
            with open(output_path, 'rb') as f:
                video_bytes.write(f.read())
            
            video_bytes.seek(0)
            return video_bytes
            
        except Exception as e:
            raise RuntimeError(f"Video compilation failed: {str(e)}")
        finally:
            # Clean up
            try:
                if 'final_clip' in locals():
                    final_clip.close()
                for clip in clips:
                    clip.close()
            except:
                pass

    def _add_captions(self, image_array, text):
        """Improved caption positioning based on aspect ratio"""
        img = Image.fromarray(image_array)
        draw = ImageDraw.Draw(img)
        
        # Dynamic font size based on video dimensions
        base_font_size = int(self.height * 0.04)  # ~51px for 1280 height
        dynamic_font = None
        
        # Try different font options with descending priority
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Common on Linux
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "arialbd.ttf", 
            "Arial_Bold.ttf",
        ]
        
        for path in font_paths:
            try:
                dynamic_font = ImageFont.truetype(path, size=base_font_size)
                break
            except:
                continue
                
        if not dynamic_font:  # Ultimate fallback with scaling
            dynamic_font = ImageFont.load_default()
            # Scale up default font using 2x transform
            dynamic_font = dynamic_font.font_variant(size=base_font_size)
        
        # Split into shorter lines for better readability
        # Adjust words per line based on aspect ratio
        words = text.split()
        lines = []
        current_line = []
        
        # Use different words per line based on aspect ratio
        words_per_line = 4 if self.is_vertical else 5
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= words_per_line:
                lines.append(' '.join(current_line))
                current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate text block positioning
        try:
            line_height = dynamic_font.getbbox("A")[3] - dynamic_font.getbbox("A")[1]
        except TypeError:
            # Fallback for older Pillow versions
            line_height = dynamic_font.getsize("A")[1]
        
        total_height = len(lines) * (line_height + 10)
        
        # Different positioning based on aspect ratio
        if self.is_vertical:  # 9:16 - Position captions near bottom third
            y_position = self.height - total_height - int(self.height * 0.15)
        else:  # 16:9 - Position captions at bottom with padding
            y_position = self.height - total_height - int(self.height * 0.1)
        
        # Ensure y_position is within bounds
        y_position = max(10, y_position)
        
        # Draw each line
        for line in lines:
            try:
                bbox = dynamic_font.getbbox(line)
                text_width = bbox[2] - bbox[0]
            except (TypeError, AttributeError):
                # Fallback for older Pillow versions
                text_width = dynamic_font.getsize(line)[0]
            
            x_position = (self.width - text_width) // 2
            x_position = max(10, x_position)  # Ensure x is within bounds
            
            # Draw outline for better visibility
            draw.text(
                (x_position, y_position),
                line,
                font=dynamic_font,
                fill="#FFFF00",  # Text fill color
                stroke_width=3,  # Outline width
                stroke_fill="#000000"  # Outline color
            )
            y_position += line_height + 10  # Move down for the next line
        
        return np.array(img)