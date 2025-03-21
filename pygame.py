import pygame
import random  # Simulating classification
import threading
import time

# Initialize pygame
pygame.init()

# Screen dimensions and setup
screen_width, screen_height = 720, 1280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Smart Trash Can")

# Colors
WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
LIGHT_CORAL = (240, 128, 128)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BUTTON_COLOR = (100, 149, 237)
BUTTON_HOVER_COLOR = (65, 105, 225)

# Fonts
font_large = pygame.font.Font(None, 80)
font_medium = pygame.font.Font(None, 60)
font_small = pygame.font.Font(None, 40)

# Variables
bin_capacity = 10  # Default: 10 gallons
trash_count = 0
recycle_count = 0
item_volume = 0.15  # Assume each item takes 0.15 gallons
current_item = "Scanning..."
running = True

# Button dimensions
button_width, button_height = 300, 80
reset_button_rect = pygame.Rect((screen_width // 2 - button_width // 2, 800), (button_width, button_height))
settings_button_rect = pygame.Rect((screen_width // 2 - button_width // 2, 900), (button_width, button_height))

# Function to simulate classifying items continuously (Replace with actual detection)
def classify_item():
    global trash_count, recycle_count, current_item
    while running:
        time.sleep(3)  # Simulate scanning every 3 seconds
        item = random.choice(["Recyclable", "Trash"])

        if item == "Recyclable":
            recycle_count += 1
        else:
            trash_count += 1

        current_item = item

# Function to reset the counts
def reset_counts():
    global trash_count, recycle_count, current_item
    trash_count = 0
    recycle_count = 0
    current_item = "Scanning..."

# Function to render text centered on the screen
def render_text_centered(text, font, color, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(screen_width // 2, y))
    screen.blit(text_surface, text_rect)

# Function to render a button
def render_button(rect, text, font, color, hover_color, mouse_pos):
    pygame.draw.rect(screen, hover_color if rect.collidepoint(mouse_pos) else color, rect, border_radius=10)
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

# Start the classification thread
threading.Thread(target=classify_item, daemon=True).start()

# Main loop
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_click = True

    # Check button clicks
    if mouse_click:
        if reset_button_rect.collidepoint(mouse_pos):
            reset_counts()
        elif settings_button_rect.collidepoint(mouse_pos):
            bin_capacity = max(4, min(32, bin_capacity + 1))  # Example adjustment

    # Calculate percentages
    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    # Clear screen
    screen.fill(WHITE)

    # Render labels
    render_text_centered(f"Smart Trash Can", font_large, BLACK, 100)
    render_text_centered(f"Item: {current_item}", font_medium, GRAY, 250)
    render_text_centered(f"Recyclable: {recycle_percentage:.1f}%", font_medium, LIGHT_BLUE, 400)
    render_text_centered(f"Trash: {trash_percentage:.1f}%", font_medium, LIGHT_CORAL, 500)

    # Render buttons
    render_button(reset_button_rect, "Reset Counts", font_small, BUTTON_COLOR, BUTTON_HOVER_COLOR, mouse_pos)
    render_button(settings_button_rect, "Adjust Bin Capacity", font_small, BUTTON_COLOR, BUTTON_HOVER_COLOR, mouse_pos)

    # Render current bin capacity
    render_text_centered(f"Current Capacity: {bin_capacity} gallons", font_small, BLACK, 1000)

    # Update display
    pygame.display.flip()

# Quit pygame
pygame.quit()
