from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create a new image with white background
    width = 800
    height = 1200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Load a font (you may need to adjust the path)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Read the sample report
    with open('sample_report.txt', 'r') as f:
        lines = f.readlines()
    
    # Draw the text
    y = 50
    for i, line in enumerate(lines):
        if i == 0:  # Title
            draw.text((50, y), line.strip(), fill='black', font=title_font)
            y += 40
        else:
            draw.text((50, y), line.strip(), fill='black', font=font)
            y += 30
    
    # Save the image
    image.save('test_report.png')
    print("Test image created: test_report.png")

if __name__ == "__main__":
    create_test_image() 