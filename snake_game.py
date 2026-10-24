
import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 700
CELL_SIZE = 10

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]

def create_food():
    """Generate food at a random grid position."""
    while True:
        pos = [
            random.randrange(0, WIDTH // CELL_SIZE) * CELL_SIZE,
            random.randrange(0, HEIGHT // CELL_SIZE) * CELL_SIZE
        ]
        if pos not in snake_body:
            return pos

food_pos = create_food()

direction = "RIGHT"
change_to = direction
score = 0

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
game_over = False
running = True

# Main game loop
while running:

    # 1. Handle events and keyboard input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN and not game_over:
            if event.key == pygame.K_UP and direction != "DOWN":
                change_to = "UP"
            elif event.key == pygame.K_DOWN and direction != "UP":
                change_to = "DOWN"
            elif event.key == pygame.K_LEFT and direction != "RIGHT":
                change_to = "LEFT"
            elif event.key == pygame.K_RIGHT and direction != "LEFT":
                change_to = "RIGHT"

            elif event.key == pygame.K_r and game_over:
                # Restart game
                snake_pos = [100, 50]
                snake_body = [[100, 50], [90, 50], [80, 50]]
                direction = "RIGHT"
                change_to = direction
                score = 0
                food_pos = create_food()
                game_over = False

    # 2. Update the game only when not over
    if not game_over:

        # Update direction
        direction = change_to

        # Move snake head
        if direction == "UP":
            snake_pos[1] -= CELL_SIZE
        elif direction == "DOWN":
            snake_pos[1] += CELL_SIZE
        elif direction == "LEFT":
            snake_pos[0] -= CELL_SIZE
        elif direction == "RIGHT":
            snake_pos[0] += CELL_SIZE

        # Add new head to body
        snake_body.insert(0, list(snake_pos))

        # Eat food and grow
        if snake_pos == food_pos:
            score += 1
            food_pos = create_food()
        else:
            snake_body.pop()

        # 3. Check wall collision
        if (
            snake_pos[0] < 0
            or snake_pos[0] >= WIDTH
            or snake_pos[1] < 0
            or snake_pos[1] >= HEIGHT
        ):
            game_over = True

        # 4. Check collision with own body
        if snake_pos in snake_body[1:]:
            game_over = True

    # 5. Draw everything
    screen.fill(BLACK)

    # Draw snake
    for pos in snake_body:
        pygame.draw.rect(
            screen,
            GREEN,
            pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE)
        )

    # Draw food once per frame
    pygame.draw.rect(
        screen,
        RED,
        pygame.Rect(food_pos[0], food_pos[1], CELL_SIZE, CELL_SIZE)
    )

    # Display score
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Display game-over message
    if game_over:
        message = font.render(
            "Game Over! Press R to Restart or close the window",
            True,
            RED
        )
        screen.blit(
            message,
            message.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        )

    # Update display and control speed
    pygame.display.flip()
    clock.tick(15)

# Quit Pygame
pygame.quit()
