import pygame
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 8
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
GOAL = (7, 7)
OBSTACLES = [
    (0,1), (1,1), (2,1), (4,1), (4,2),
    (3,2), (3,4), (3,5), (4,5), (5,5),
    (5,4), (6,6), (7,6), (1,3), (2,4),
    (0,5), (2,6), (0,7)]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

portal = (7, 0)  # Ponto de teletransporte
return_portal = (1, 7)  # Ponto de retorno

# Parâmetros do Q-Learning
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 100
MAX_STEPS = 100
BETA = 5

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BLUE = (50, 50, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)

# Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Funções auxiliares
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_next_state(state, action):
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    if next_state == portal:
        return return_portal
    elif is_valid(next_state):
        return next_state
    else:
        return state

def get_reward(state):
    if state == GOAL:
        return 100
    elif state == return_portal:
        return  50
    elif state in OBSTACLES:
        return -10
    else:
        return -1
    
def potential(state):
    return - (abs(state[0] - GOAL[0]) + abs(state[1] - GOAL[1]))

def get_shaped_reward(state, next_state, visits):
    base_reward = get_reward(next_state)
    shaping = GAMMA * potential(next_state) - potential(state)
    revisit_penalty = - BETA * visits[next_state[0], next_state[1]] if visits[next_state[0], next_state[1]] > 0 else 0
    return base_reward + shaping + revisit_penalty

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) == START:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            elif (x, y) == portal:
                color = YELLOW
            elif (x, y) == return_portal:
                color = YELLOW
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

def draw_agent(screen, pos, color=BLUE):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, color, rect)

def reward_function(state):
    if state == GOAL:
        return 100
    elif state == return_portal:
        return 50
    elif state in OBSTACLES:
        return -10
    
    return reward_function(state)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -----------------------------
# TREINAMENTO COM VISUALIZAÇÃO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    print(f"Treinando episódio {episode + 1}/{EPISODES}")
    state = START
    visits = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    visits[state[0], state[1]] += 1

    for step in range(MAX_STEPS):
        process_events()

        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state = get_next_state(state, action)
        reward = get_shaped_reward(state, next_state, visits)

        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = next_state
        visits[state[0], state[1]] += 1

        # Visualização do treinamento
        screen.fill(WHITE)
        draw_grid(screen)
        draw_agent(screen, state, ORANGE)
        pygame.display.flip()
        clock.tick(60)

        if state == GOAL:
            break

# -----------------------------
# EXECUÇÃO DO AGENTE TREINADO
# -----------------------------
pygame.display.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Execução do Agente Treinado")
clock = pygame.time.Clock()

agent_pos = START
path = [agent_pos]
reached_goal = False
running = True

while running:
    process_events()

    screen.fill(WHITE)
    draw_grid(screen)

    # Desenha caminho já percorrido
    for pos in path:
        draw_agent(screen, pos, BLUE)

    # Desenha agente atual
    if not reached_goal:
        draw_agent(screen, agent_pos, BLUE)

    pygame.display.flip()
    clock.tick(30)  # FPS alto para manter janela fluida

    if not reached_goal:
        if agent_pos != GOAL:
            action = np.argmax(q_table[agent_pos[0], agent_pos[1]])
            next_pos = get_next_state(agent_pos, action)
            if next_pos == agent_pos:
                print("Agente está preso!")
                reached_goal = True
            else:
                path.append(next_pos)
                agent_pos = next_pos
                time.sleep(0.8)  # <-- controle de velocidade da execução
        else:
            reached_goal = True
            print("\nCaminho percorrido pelo agente:")
            print(path)
