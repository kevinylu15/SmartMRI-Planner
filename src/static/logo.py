"""
Generate a logo for SmartMRI Planner
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with white background
    width, height = 400, 200
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw a brain outline
    brain_color = (65, 105, 225)  # Royal blue
    
    # Simplified brain shape
    brain_x = 100
    brain_y = 100
    brain_width = 100
    brain_height = 80
    
    # Draw brain lobes
    draw.ellipse((brain_x - brain_width/2, brain_y - brain_height/2, 
                  brain_x + brain_width/2, brain_y + brain_height/2), 
                 outline=brain_color, width=3)
    
    # Draw some "brain waves" to represent MRI
    wave_color = (70, 130, 180)  # Steel blue
    wave_points = []
    wave_start_x = brain_x + brain_width/2 + 10
    wave_y = brain_y
    wave_width = 80
    wave_height = 30
    
    for i in range(0, 81, 10):
        x = wave_start_x + i
        if (i // 10) % 2 == 0:
            y = wave_y - wave_height/2
        else:
            y = wave_y + wave_height/2
        wave_points.append((x, y))
    
    # Connect the points with lines
    for i in range(len(wave_points) - 1):
        draw.line([wave_points[i], wave_points[i+1]], fill=wave_color, width=3)
    
    # Add text
    try:
        # Try to use a nice font if available
        font = ImageFont.truetype("Arial.ttf", 40)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Draw text
    text_color = (25, 25, 112)  # Midnight blue
    draw.text((width/2, height/2), "SmartMRI", fill=text_color, font=font, anchor="mm")
    
    # Save the image
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    image.save(logo_path)
    print(f"Logo created at {logo_path}")
    return logo_path

if __name__ == "__main__":
    create_logo()
