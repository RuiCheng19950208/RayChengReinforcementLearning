import pygame
import sys
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import copy
import heapq
# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Breakout Game with DQN Agent")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PURPLE = (128, 0, 128)
BRICK_COLORS = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, PURPLE]

# Game variables
clock = pygame.time.Clock()
FPS = 60
font = pygame.font.SysFont('Arial', 26)
large_font = pygame.font.SysFont('Arial', 40, bold=True)

# Paddle properties
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_SPEED = 8

# Ball properties
BALL_RADIUS = 10
BALL_SPEED = 6

# Brick properties
BRICK_WIDTH = 80
BRICK_HEIGHT = 30
BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_NUM = BRICK_ROWS * BRICK_COLS
BRICK_GAP = 5

STATE_SIZE = 5 + (BRICK_NUM * 3)
# STATE_SIZE = 5 
NUM_EPISODES = 5000
SWITCH_EPISODE = 300

# UPDATE_TARGET_EVERY = 10  # Update target network every N episodes
SAVE_MODEL_EVERY = 100    # Save model every N episodes
RENDER_EVERY = 50
BALL_COLLISION_COOLDOWN = 5
BATCH_SIZE = 128
HISTORY_LENGTH = 5 # Size of the frames we put into the NN
REWARD_SCALING_FACTOR = 1/10

EPSILON_THRESHOLD_LOW = -400 # Once the score reached this threshold, epsilon will be set to 1
EPSILON_THRESHOLD_UP = 200 # Once the score reached this threshold, epsilon will be set to EPSILON_MIN
EPSILON_MIN = 0.01
# EPSILON_DECAY = 0.98
GAMMA = 0.9
NETWORK_UPDATE_EVERY = 5000
MEMORY_CAPACITY = 10000
BEST_MEMORY_CAPACITY = 10000
WORST_MEMORY_CAPACITY = 10000
# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Global Variables(Have to update when we restart)
PREV_BRICK_COUNT = BRICK_NUM


#Brick Reward
BRICK_DESTRUCTION_REWARD = 1
BRICK_DESTRUCTION_REWARD_FACTOR = 10 # If remaining bricks number are low, reward will be high, last brick will be the highest as this value
BRICK_SUCCESSION_REWARD = 3

# Paddle Reward
HIT_PADDLE_REWARD = 20
ALIGMENT_REWARD = 30
NOT_HIT_BALL_IN_THIS_ROUND_FACTOR = 0.3
# Life Reward
LOST_LIFE_BASE_PUNISHMENT = 30
LOST_LIFE_DISTANCE_PUNISHMENT = 100 # Max value
TIME_STEP_PUNISHMENT = 0.1

# Win Reward
WIN_REWARD = 2000


class Paddle:
    def __init__(self):
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.x = (WIDTH - self.width) // 2
        self.y = HEIGHT - self.height - 20
        self.speed = PADDLE_SPEED
        self.color = WHITE
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    
    def update(self, action):
        # Action: 0 = left, 1 = stay, 2 = right
        if action == 0 and self.x > 0 - self.width:  # Move left
            self.x -= self.speed
        elif action == 2 and self.x < WIDTH + self.width:  # Move right
            self.x += self.speed
        
        # Update rect position for collision detection
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

class Ball:
    def __init__(self):
        self.radius = BALL_RADIUS
        self.reset()
        self.ball_collision_cooldown = 0
        # Add collision state flags
        self.is_paddle_hit = False
        self.is_brick_hit = False
        self.hit_brick = None  # Store the hit brick
        self.ball_lost = False
        self.brick_succession_count = 0
        self.hit_paddle_reward_factor = 1
    
    def reset(self):
        # Position ball on top of paddle
        self.x = WIDTH // 2
        self.y = HEIGHT - PADDLE_HEIGHT - self.radius - 30
        
        # Set random angle between 210 and 330 degrees (downward with variation)
        angle = random.randint(210,330) * math.pi / 180
        self.dx = math.cos(angle) * BALL_SPEED
        self.dy = math.sin(angle) * BALL_SPEED
        # Reset collision states
        self.is_paddle_hit = False
        self.is_brick_hit = False
        self.hit_brick = None
        self.ball_lost = False
        self.brick_succession_count = 0
        self.hit_paddle_reward_factor = 1

    def update(self, paddle, bricks):
        # Reset collision states at the beginning of each frame
        self.is_paddle_hit = False
        self.is_brick_hit = False
        self.hit_brick = None
        self.ball_lost = False
        
        # Move the ball
        self.x += self.dx
        self.y += self.dy

        # Reduce cooldown timer
        if self.ball_collision_cooldown > 0:
            self.ball_collision_cooldown -= 1

        # Check if ball is below paddle (lost)
        if self.y >= HEIGHT:
            self.ball_lost = True

        # Wall collision
        if self.x <= self.radius:
            self.dx = abs(self.dx)
        if self.x >= WIDTH - self.radius:
            self.dx  = -abs(self.dx)
        if self.y <= self.radius:
            self.dy = abs(self.dy)
            
        if self.ball_collision_cooldown == 0:
            # Paddle collision
            if (
                self.y + self.radius >= paddle.y and 
                self.y - self.radius <= paddle.y + paddle.height and
                self.x >= paddle.x and 
                self.x <= paddle.x + paddle.width):
                
                # Calculate bounce angle based on where ball hits paddle
                relative_intersect_x = (paddle.x + (paddle.width / 2)) - self.x
                normalized_intersect_x = relative_intersect_x / (paddle.width / 2)
                bounce_angle = normalized_intersect_x * (math.pi / 3)  # Max 60 degrees
                
                # Set new velocity
                self.dx = BALL_SPEED * -math.sin(bounce_angle)
                self.dy = BALL_SPEED * -math.cos(bounce_angle)
                
                # Set collision cooldown and paddle collision flag
                self.ball_collision_cooldown = BALL_COLLISION_COOLDOWN
                self.is_paddle_hit = True

                if self.brick_succession_count ==0:
                    self.hit_paddle_reward_factor *= NOT_HIT_BALL_IN_THIS_ROUND_FACTOR
                else:
                    self.hit_paddle_reward_factor = 1

                self.brick_succession_count = 0
                    # Check brick collision

            for row in bricks:
                for brick in row:
                    if brick.visible and self.check_brick_collision(brick):
                        self.is_brick_hit = True
                        self.hit_brick = brick
                        self.brick_succession_count +=1
                        self.ball_collision_cooldown = BALL_COLLISION_COOLDOWN
                        break
                if self.is_brick_hit:
                    break 
        

        
        
    
    def check_brick_collision(self, brick):
        if not brick.visible:
            return False
        
        # Get the ball's bounding box
        ball_left = self.x - self.radius
        ball_right = self.x + self.radius
        ball_top = self.y - self.radius
        ball_bottom = self.y + self.radius
        
        # First do a rough check (rectangle intersection)
        if (ball_right < brick.rect.left or 
            ball_left > brick.rect.right or 
            ball_bottom < brick.rect.top or 
            ball_top > brick.rect.bottom):
            return False
        
        # Calculate the closest point from the ball center to the brick
        closest_x = max(brick.rect.left, min(self.x, brick.rect.right))
        closest_y = max(brick.rect.top, min(self.y, brick.rect.bottom))
        
        # Calculate distance
        distance = math.sqrt((self.x - closest_x) ** 2 + (self.y - closest_y) ** 2)
        
        if distance <= self.radius:
            
            # Determine the main collision face
            # Calculate the distance from the ball center to each face of the brick
            dist_left = abs(brick.rect.left - ball_right)
            dist_right = abs(brick.rect.right - ball_left)
            dist_top = abs(brick.rect.top - ball_bottom)
            dist_bottom = abs(brick.rect.bottom - ball_top)
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
            # Determine bounce direction based on the closest face
            if min_dist == dist_left or min_dist == dist_right:
                self.dx *= -1  # Horizontal bounce
            else:
                self.dy *= -1  # Vertical bounce
                
            brick.visible = False
            return True
    
        return False
    
    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)

class Brick:
    def __init__(self, x, y, color):
        self.width = BRICK_WIDTH
        self.height = BRICK_HEIGHT
        self.x = x
        self.y = y
        self.color = color
        self.visible = True
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self):
        if self.visible:
            pygame.draw.rect(screen, self.color, self.rect)
            # Draw brick outline
            pygame.draw.rect(screen, BLACK, self.rect, 2)

def create_bricks():
    bricks = []
    for row in range(BRICK_ROWS):
        brick_row = []
        for col in range(BRICK_COLS):
            x = col * (BRICK_WIDTH + BRICK_GAP) + BRICK_GAP
            y = row * (BRICK_HEIGHT + BRICK_GAP) + BRICK_GAP + 50  # Start from y=50
            brick = Brick(x, y, BRICK_COLORS[row % len(BRICK_COLORS)])
            brick_row.append(brick)
        bricks.append(brick_row)
    return bricks

def check_game_won(bricks):
    # Check if all bricks are destroyed
    for row in bricks:
        for brick in row:
            if brick.visible:
                return False
    return True

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, ball_state_dim=5, brick_state_dim= 3*BRICK_NUM, history_length=HISTORY_LENGTH, action_size=3):
        super(DQN, self).__init__()
        
        # Ball handling branch (5 channels, 10 frames)
        self.ball_conv1 = nn.Conv1d(ball_state_dim, 32, kernel_size=3, stride=1)  # Output: [32, 8]
        self.ball_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)             # Output: [64, 6]
        ball_flat_size = 64 * 6  # Size after flattening
        
        # Brick branch (135 channels, 10 frames)
        self.brick_conv1 = nn.Conv1d(brick_state_dim, 32, kernel_size=3, stride=1)  # Output: [32, 8]
        self.brick_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)              # Output: [64, 6]
        brick_flat_size = 64 * 6  # Size after flattening
        
        # Fusion layer
        self.combine_fc = nn.Linear(ball_flat_size + brick_flat_size, 128)
        self.output_fc = nn.Linear(128, action_size)
        
        # Initialization
        # Initialize brick branch to 0
        nn.init.zeros_(self.brick_conv1.weight)
        nn.init.zeros_(self.brick_conv1.bias)
        nn.init.zeros_(self.brick_conv2.weight)
        nn.init.zeros_(self.brick_conv2.bias)
        
        # Initialize ball branch and fusion layer with Kaiming initialization
        nn.init.kaiming_uniform_(self.ball_conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.ball_conv1.bias)
        nn.init.kaiming_uniform_(self.ball_conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.ball_conv2.bias)
        nn.init.kaiming_uniform_(self.combine_fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.combine_fc.bias)
        nn.init.kaiming_uniform_(self.output_fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.output_fc.bias)

    def forward(self, x):
        # Split ball and brick states
        ball_paddle = x[:, :5, :]    # [batch_size, 5, 10]
        bricks = x[:, 5:, :]         # [batch_size, 135, 10]
        
        # Ball branch
        bp = F.leaky_relu(self.ball_conv1(ball_paddle), negative_slope=0.01)  # [batch_size, 32, 8]
        bp = F.leaky_relu(self.ball_conv2(bp), negative_slope=0.01)           # [batch_size, 64, 6]
        bp = bp.view(bp.size(0), -1)                                          # [batch_size, 64*6 = 384]
        
        # Brick branch
        br = F.leaky_relu(self.brick_conv1(bricks), negative_slope=0.01)      # [batch_size, 32, 8]
        br = F.leaky_relu(self.brick_conv2(br), negative_slope=0.01)          # [batch_size, 64, 6]
        br = br.view(br.size(0), -1)                                          # [batch_size, 64*6 = 384]
        
        # Fusion
        combined = torch.cat((bp, br), dim=1)                                 # [batch_size, 64*6 + 64*6]
        combined = F.leaky_relu(self.combine_fc(combined), negative_slope=0.01)
        output = self.output_fc(combined) # [batch_size, 3]

        return output


# Experience Replay Memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class WorstMemory:
    def __init__(self, capacity):
        self.memory = []  # tuple of (reward, counter, experience)
        self.capacity = capacity
        self.counter = 0  # Add a counter as a unique ID
    
    def push(self, *args):
        try:
            experience = Experience(*args)
            reward_scalar = float(experience.reward.item() if hasattr(experience.reward, 'item') else experience.reward[0] if hasattr(experience.reward, '__len__') else experience.reward)
            # print(f"Added new experience to worst memory: {experience.reward}")
            # If heap isn't full yet, just add it
            if len(self.memory) < self.capacity:
                # Store as tuple with reward directly for min-heap (lowest rewards)
                heapq.heappush(self.memory, (reward_scalar, self.counter, experience))
                self.counter += 1
            else:
                # Get reward of the highest reward item in our worst memory
                highest_reward = self.memory[0][0]
                
                # Only add if new experience has a lower reward (worse)
                if reward_scalar < highest_reward:
                    # Remove highest reward experience and add new one
                    heapq.heappushpop(self.memory, (reward_scalar, self.counter, experience))
                    self.counter += 1
            # print(f"Added new experience to worst memory: {experience.reward}")
        except Exception as e:
            print(f"Error in WorstMemory.push: {e}")
                
    
    def sample(self, batch_size):
        experiences = [item[2] for item in random.sample(self.memory, batch_size)]
        return experiences
    
    def __len__(self):
        return len(self.memory)

class BestMemory:
    def __init__(self, capacity):
        self.memory = [] # tuple of (reward, counter, experience)
        self.capacity = capacity
        self.counter = 0  # Add a counter as a unique ID
    
    def push(self, *args):
        try:
            experience = Experience(*args)
            reward_scalar = float(experience.reward.item() if hasattr(experience.reward, 'item') else experience.reward[0] if hasattr(experience.reward, '__len__') else experience.reward)
            
            # If heap isn't full yet, just add it
            if len(self.memory) < self.capacity:
                # Add counter as the second element
                heapq.heappush(self.memory, (-reward_scalar, self.counter, experience))
                self.counter += 1
            else:
                # Get reward of the lowest priority item
                lowest_reward = -self.memory[0][0]
                
                # Only add if new experience has a higher reward
                if reward_scalar > lowest_reward:
                    # Remove lowest reward experience and add new one
                    heapq.heappushpop(self.memory, (-reward_scalar, self.counter, experience))
                    self.counter += 1
        except Exception as e:
            print(f"Error in BestMemory.push: {e}")
            # Skip this experience and continue
    
    def sample(self, batch_size):
        experiences = [item[2] for item in random.sample(self.memory, batch_size)]
        return experiences
    
    def __len__(self):
        return len(self.memory)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, action_size =3, lr=0.001):
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate

        self.batch_size = BATCH_SIZE
        
        # Networks and optimizer
        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.normal_memory = ReplayMemory(MEMORY_CAPACITY)
        self.is_paddle_hit_memory = BestMemory(BEST_MEMORY_CAPACITY)
        self.is_paddle_miss_memory = WorstMemory(WORST_MEMORY_CAPACITY)
        
        # For tracking
        # self.loss_history = []

    def freeze_bricks_branch(self):
        # Freeze bricks branch parameters
        for param in self.policy_net.brick_conv1.parameters():
            param.requires_grad = False
        for param in self.policy_net.brick_conv2.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        # Unfreeze all parameters
        for param in self.policy_net.parameters():
            param.requires_grad = True
        
    def act(self, state, train=True):
        # Epsilon-greedy action selection
        if train and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def learn(self,episode):
        if len(self.normal_memory) < self.batch_size:
            return
        
        # Sampling
        is_paddle_hit_samples = self.is_paddle_hit_memory.sample(min(3, len(self.is_paddle_hit_memory)))
        is_paddle_miss_samples = self.is_paddle_miss_memory.sample(min(3, len(self.is_paddle_miss_memory)))
        normal_samples = self.normal_memory.sample(self.batch_size - len(is_paddle_hit_samples) - len(is_paddle_miss_samples))
        merged_experiences = is_paddle_miss_samples + is_paddle_hit_samples + normal_samples
        
        # Unpack
        states, actions, rewards, next_states, dones = zip(*merged_experiences)
        
        # Convert to batch tensors
        states = torch.cat(states, dim=0)           # [batch_size, 140, 10], compressed from [batch_size, 1, 140, 10]
        actions = torch.LongTensor(actions).to(DEVICE).unsqueeze(1)  # [batch_size, 1]
        rewards = torch.FloatTensor(rewards).to(DEVICE)             # [batch_size]
        next_states = torch.cat(next_states, dim=0)  # [batch_size, 140, 10]
        dones = torch.FloatTensor(dones).to(DEVICE)                 # [batch_size]
        
        # Debug print
        # print("States shape:", states.shape)  # Should be [batch_size, 140, 10]
        # print("Actions shape:", actions.shape)  # Should be [batch_size, 1]
        
        # Calculate Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze(1) # [batch_size] select the action which is taken and remove the dimension at index 1
        
        # Calculate target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]  # [batch_size]
        # print("next_q shape:",next_q.shape)

        # Calculate expected Q values
        target_q = REWARD_SCALING_FACTOR * rewards + (self.gamma * next_q * (1 - dones))  # [batch_size]
        # print("target_q shape:",target_q.shape)
        # print("current_q:",current_q)
        # print("target_q:",target_q)
        

        # Calculate loss and update network
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.01 + max(1.99*(2000-episode)/2000,0))
        self.optimizer.step()
        
        
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())



def calculate_reward(paddle, ball, bricks):
    global PREV_BRICK_COUNT
    reward = 0
    # 1. Block hitting reward (main goal)
    current_brick_count = sum(1 for row in bricks for brick in row if brick.visible)
    
    # Reward for reducing the number of blocks (encouraging block destruction)
    if current_brick_count < PREV_BRICK_COUNT:

        reward += BRICK_DESTRUCTION_REWARD # Base reward
        reward += ((BRICK_NUM -current_brick_count) / BRICK_NUM) * BRICK_DESTRUCTION_REWARD_FACTOR
        reward += ball.brick_succession_count * BRICK_SUCCESSION_REWARD
        
    PREV_BRICK_COUNT = current_brick_count
    
    # 2. Calculate trajectory prediction reward only at the moment of catching the ball
    if ball.is_paddle_hit:
        # Basic reward for catching the ball (maintaining basic skills)
        reward += HIT_PADDLE_REWARD * ball.hit_paddle_reward_factor

        # Calculate the closest visible brick
        closest_brick = None
        min_distance = float('inf')
        for row in bricks:
            for brick in row:
                if brick.visible:
                    distance = abs(ball.x - (brick.x + brick.width/2)) + abs(ball.y - (brick.y + brick.height/2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_brick = brick
        
        if closest_brick:
            # Reward ball trajectory that approaches bricks
            target_dx = (closest_brick.x + closest_brick.width/2 - ball.x) 
            target_dy = (closest_brick.y + closest_brick.height/2 - ball.y) 
            # Calculate the amplitude of this vector
            amplitude = math.sqrt(target_dx**2 + target_dy**2)
            target_dx /= amplitude
            target_dy /= amplitude
            
            # Calculate the cosine of the angle between the target direction and the ball's velocity
            dot_product = target_dx * (ball.dx/BALL_SPEED) + target_dy * (ball.dy/BALL_SPEED)
            
            
            reward += ALIGMENT_REWARD * dot_product * ball.hit_paddle_reward_factor
    
    

    # 5. Ball loss penalty
    if ball.ball_lost:
        reward -= LOST_LIFE_BASE_PUNISHMENT + LOST_LIFE_DISTANCE_PUNISHMENT * abs(ball.x - paddle.x + paddle.width/2)/(WIDTH+PADDLE_WIDTH)
    
    # 6. Game end reward
    if current_brick_count == 0:
        reward += WIN_REWARD
    
    # 7. Survival penalty (to prevent excessive stalling)
    reward -= TIME_STEP_PUNISHMENT

    # 8. Reward for getting away from the paddle
    if paddle.x - ball.x > 0 or paddle.x - ball.x < -PADDLE_WIDTH: 
        reward -= 10 * abs(paddle.x - ball.x + PADDLE_WIDTH/2)/WIDTH
    else:
        reward += TIME_STEP_PUNISHMENT

    
    return reward


class GameStateBuffer:
    def __init__(self, history_length=10, state_dim=140):
        self.history_length = history_length
        self.state_dim = state_dim
        self.buffer = deque(maxlen=history_length)
        self.reset()
    
    def reset(self):
        zero_state = torch.zeros(self.state_dim, dtype=torch.float32).to(DEVICE)  # [140]
        for _ in range(self.history_length):
            self.buffer.append(zero_state)
    
    def append(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(DEVICE)  # [140]
        self.buffer.append(state)
    
    def get_state(self):
        # Return [1, 140, 10] tensor
        state = torch.stack(list(self.buffer), dim=1)  # [140, 10]
        return state.unsqueeze(0)  # [1, 140, 10]

# Extract state from game elements
def get_state(paddle, ball, bricks):
    # Normalize positions to [0, 1]
    paddle_x = (paddle.x+PADDLE_WIDTH) / (WIDTH + PADDLE_WIDTH)
    ball_x = (ball.x) / (WIDTH)
    ball_y = ball.y / HEIGHT
    ball_dx_norm = ball.dx / BALL_SPEED
    ball_dy_norm = ball.dy / BALL_SPEED
    
    # Get brick states (0 = destroyed, 1 = visible)
    brick_states = []
    for row in bricks:
        for brick in row:
            # Include normalized positions and visibility
            brick_states.extend([brick.x / WIDTH, brick.y / HEIGHT, 1.0 if brick.visible else 0.0])
    
    # Combine all states
    state = [paddle_x, ball_x, ball_y,ball_dx_norm,ball_dy_norm] + brick_states

    return torch.FloatTensor(state).to(DEVICE) # [140]

# Training function
def train_agent(num_episodes=1000, render=True):
    # Initialize agent
    # State size: paddle_x, ball_x, ball_y, plus 3 values per brick (x, y, visible)

    agent = DQNAgent()
    agent.freeze_bricks_branch()

    # Game state buffer
    buffer = GameStateBuffer()


    # Training stats
    scores = deque(maxlen=50)
    accumulated_step = 0
    
    for episode in range(1, num_episodes + 1):
        # Initialize game
        paddle = Paddle()
        ball = Ball()
        bricks = create_bricks()
        lives = 3
        score = 0

        initial_state = get_state(paddle, ball, bricks) #[140]
        buffer.reset()
        buffer.append(initial_state)  # Fill with initial frame
        state = buffer.get_state()    # [1, 140, 10]
        # print("state shape:",state.shape)
        
        # Game loop
        done = False
        steps = 0
        PREV_BRICK_COUNT = BRICK_NUM

        if episode == SWITCH_EPISODE:
            agent.unfreeze_all()
            print("Unfreeze all parameters")

        while not done:
            steps += 1
            accumulated_step+=1
            
            # Get action from agent
            

            
            action = agent.act(state)  # Need to adjust agent.act to support tensor input
            # Apply action to game
            paddle.update(action)
            ball.update(paddle, bricks)

            # print(ball.x,paddle.x)

            reward = calculate_reward(paddle, ball, bricks)



            if ball.ball_lost:
                lives -= 1
                is_ball_lost = True
                ball.reset()
                if lives <= 0:
                    done = True
            else:
                is_ball_lost =False
            
            
            
            if check_game_won(bricks):
                done = True
            
            # Get new state
            next_state_single = get_state(paddle, ball, bricks) #[140]
            # print("next_state_single shape:",next_state_single.shape)
            buffer.append(next_state_single)
            next_state = buffer.get_state()  # [1,140, 10]
            # print("next_state shape:",next_state.shape)

            if is_ball_lost:
                agent.is_paddle_miss_memory.push(state, action, reward, next_state, done)

            # Store experience when paddle is hit
            if ball.is_paddle_hit:
                agent.is_paddle_hit_memory.push(state, action, reward, next_state, done)
            
            # Store experience
            agent.normal_memory.push(state, action, reward, next_state, done)

            # Learn from experience
            agent.learn(episode)
            
            # Update state
            state = next_state
            score += reward

            if accumulated_step % NETWORK_UPDATE_EVERY == 0:
                agent.update_target_net()
                accumulated_step = 0
                print(f"Target network updated")
            
            # Render game if requested
            if render and episode % RENDER_EVERY == 0:  # Render every 100 episodes
                screen.fill(BLACK)
                paddle.draw()
                ball.draw()
                agent.epsilon = 0
                
                for row in bricks:
                    for brick in row:
                        brick.draw()
                
                # Display info
                episode_text = font.render(f"Episode: {episode}/{num_episodes}", True, WHITE)
                lives_text = font.render(f"Lives: {lives}", True, WHITE)
                score_text = font.render(f"Score: {score:.1f}", True, WHITE)
                eps_text = font.render(f"Epsilon: {agent.epsilon:.2f}", True, WHITE)
                
                screen.blit(episode_text, (20, 10))
                screen.blit(lives_text, (WIDTH - lives_text.get_width() - 20, 10))
                screen.blit(score_text, (20, 40))
                screen.blit(eps_text, (WIDTH - eps_text.get_width() - 20, 40))
                
                pygame.display.flip()
                
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                # Cap framerate
                clock.tick(FPS)
        
        # Epsilon should be defined by current score
        # If score is low, epsilon should be high (encourage exploration)
        # If score is high, epsilon should be low (encourage keeping the current policy)


        # Episode complete, log stats
        scores.append(score)
        avg_score = np.mean(scores)
        
        print(f"Episode: {episode}/{num_episodes}, Score: {score:.1f}, Avg Score: {avg_score:.1f}, Epsilon: {agent.epsilon:.2f}, Steps: {steps}","paddle_hit_memory:",len(agent.is_paddle_hit_memory),"paddle_miss_memory:",len(agent.is_paddle_miss_memory))
        
        if render and episode % RENDER_EVERY == 0:
            agent.epsilon = 0 #Make sure the agent is not exploring
        else:
            max_epsilon = 1 - (min(episode, 100) / 100) * 0.9
            if avg_score < EPSILON_THRESHOLD_LOW:
                agent.epsilon = max_epsilon
            elif avg_score > EPSILON_THRESHOLD_UP:
                agent.epsilon = EPSILON_MIN
            else:
                agent.epsilon = EPSILON_MIN + ((max_epsilon - EPSILON_MIN) * (EPSILON_THRESHOLD_UP-avg_score) / (EPSILON_THRESHOLD_UP - EPSILON_THRESHOLD_LOW))
        # Update target network
        # if episode % UPDATE_TARGET_EVERY == 0:
        #     agent.update_target_net()
        #     print(f"Target network updated at episode {episode}")
        
        
        # Save model
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(f"breakout_dqn_ep{episode}.pth")
            print(f"Model saved at episode {episode}")
    
    # Save final model
    agent.save("breakout_dqn_final.pth")
    print("Training complete!")
    return agent

# Play game with trained agent
def play_with_agent(agent, num_games=5):
    for game in range(num_games):
        print(f"Starting game {game+1}/{num_games}")
        
        # Initialize game
        paddle = Paddle()
        ball = Ball()
        bricks = create_bricks()
        lives = 3
        score = 0
        start_time = time.time()
        
        # Game loop
        game_over = False

        initial_state = get_state(paddle, ball, bricks) #[140]
        buffer = GameStateBuffer()
        buffer.reset()
        buffer.append(initial_state)  # Fill with initial frame
        state = buffer.get_state()    # [1, 140, 10]
        
        while not game_over:
            # Get state and action

            next_state_single = get_state(paddle, ball, bricks) #[140]
            # print("next_state_single shape:",next_state_single.shape)
            buffer.append(next_state_single)
            next_state = buffer.get_state()  # [1,140, 10]
            action = agent.act(next_state, train=False)  # No exploration
            

            # Apply action to game
            paddle.update(action)
            ball.update(paddle, bricks)

            # print(ball.x,paddle.x)
            
            # Check if game is over (out of lives or all bricks destroyed)
            

            reward = calculate_reward(paddle, ball, bricks)

            if ball.ball_lost:
                lives -= 1
                ball.reset()
                if lives <= 0:
                    done = True
            
            if check_game_won(bricks):
                game_over = True
            
            score += reward
            
            # Render
            screen.fill(BLACK)
            paddle.draw()
            ball.draw()
            
            for row in bricks:
                for brick in row:
                    brick.draw()
            
            # Display info
            game_text = font.render(f"Game: {game+1}/{num_games}", True, WHITE)
            lives_text = font.render(f"Lives: {lives}", True, WHITE)
            score_text = font.render(f"Score: {score:.1f}", True, WHITE)
            
            screen.blit(game_text, (20, 10))
            screen.blit(lives_text, (WIDTH - lives_text.get_width() - 20, 10))
            screen.blit(score_text, (20, 40))
            
            pygame.display.flip()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Cap framerate
            clock.tick(FPS)
        
        # Game over
        elapsed_time = time.time() - start_time
        print(f"Game {game+1} complete. Score: {score:.1f}, Time: {elapsed_time:.1f}s")

def main():
    # Choose mode
    train_mode = True  # Set to False to play with a trained agent
    
    if train_mode:
        # Train agent
        agent = train_agent(num_episodes=NUM_EPISODES, render=True)
    else:
        # Load pre-trained agent

        agent = DQNAgent()
        # agent.load("breakout_dqn_final.pth")
        agent.load("breakout_dqn_ep300.pth")
        agent.epsilon = 0  # No exploration for evaluation
        
        # Play with trained agent
        play_with_agent(agent)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()