import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation
from tetris_gymnasium.envs.tetris import Tetris

# importing webdriver from selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
from time import sleep

from dqn import DQN
from replay_buffer import ReplayBuffer

class TetrisAgent:
    def __init__(self, gamma=0.99, batch_size=64, target_update_freq=1000, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.9999, global_step=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DQN().to(self.device)
        self.target = DQN().to(self.device)

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.global_step = global_step

        self.wrapped_env = Tetris(render_mode="human")
        self.wrapped_env = GroupedActionsObservations(
            self.wrapped_env, observation_wrappers=[FeatureVectorObservation(self.wrapped_env)]
        )

    def train(self, num_episodes=5000):

        for episode in range(self.global_step, num_episodes + self.global_step):
            obs = self.wrapped_env.reset()[0]  # shape: (40, 13)
            done = False
            episode_reward = 0
            lines_cleared = 0

            while not done:
                self.wrapped_env.render()
                cv2.waitKey(1)
                # epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(40)
                else:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 40, 13)
                        q_values = self.policy(obs_tensor)  # (1, 40)
                        action = q_values.argmax(dim=1).item()

                next_obs, reward, done, truncated, info = self.wrapped_env.step(action)
                
                self.buffer.push(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward += reward
                
                lines_cleared += info['lines_cleared']

                # Train the model
                if len(self.buffer) >= self.batch_size:
                    b_obs, b_action, b_reward, b_next_obs, b_done = self.buffer.sample(self.batch_size)

                    b_obs = b_obs.to(self.device)         # (batch, 40, 13)
                    b_action = b_action.to(self.device)   # (batch,)
                    b_reward = b_reward.to(self.device)   # (batch,)
                    b_next_obs = b_next_obs.to(self.device)
                    b_done = b_done.to(self.device)

                    # Get Q-values for current state
                    q_values = self.policy(b_obs)          # (batch, 40)
                    q_val = q_values.gather(1, b_action.unsqueeze(1)).squeeze(1)  # (batch,)

                    # Get target Q-values
                    with torch.no_grad():
                        next_q_vals = self.target(b_next_obs).max(dim=1)[0]  # (batch,)
                        q_target = b_reward + self.gamma * next_q_vals * (1 - b_done)

                    loss = F.mse_loss(q_val, q_target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Update target model
                if self.global_step % self.target_update_freq == 0:
                    self.target.load_state_dict(self.policy.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.global_step += 1
            print(f"Episode {episode}, Reward: {episode_reward}, Lines cleared: {lines_cleared}, Epsilon: {self.epsilon:.4f}")

    def save_checkpoint(self, checkpoint_name):
        checkpoint = {
            'policy': self.policy.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'global_step': self.global_step,
        }
        torch.save(checkpoint, checkpoint_name)

        print(f"Checkpoint saved to '{checkpoint_name}'. Epsilon: {self.epsilon}, Global Step: {self.global_step}")

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        self.policy.load_state_dict(checkpoint['policy'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded from '{checkpoint_name}'. Epsilon: {self.epsilon}, Global Step: {self.global_step}")

    def run(self, num_runs=100):
        lines_20 = 0
        avg_lines = 0
        avg_reward = 0

        for run in range(num_runs):
            obs = self.wrapped_env.reset()[0]  # shape: (40, 13)
            done = False
            run_reward = 0
            lines_cleared = 0

            while not done:
                self.wrapped_env.render()
                cv2.waitKey(1)
                
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 40, 13)
                    q_values = self.policy(obs_tensor)  # (1, 40)
                    action = q_values.argmax(dim=1).item()
                
                next_obs, reward, done, _, info = self.wrapped_env.step(action)
                
                obs = next_obs
                run_reward += reward
                lines_cleared += info['lines_cleared']

            if lines_cleared >= 20:
                lines_20 += 1
            avg_lines += lines_cleared
            avg_reward += run_reward

            print(f"Run {run}, Reward: {run_reward}, Lines Cleared: {lines_cleared}")

        print(f"Total runs with 20 lines cleared: {lines_20} out of {num_runs}")
        print(f"Average reward per run: {avg_reward / num_runs:.2f}")
        print(f"Average lines cleared per run: {avg_lines / num_runs:.2f}")

    def play(self):
        # Here Chrome will be used
        driver = webdriver.Chrome()
        
        # URL of website
        url = "https://jstris.jezevec10.com/?play=1&mode=2"
        
        # Opening the website
        driver.get(url)
        actions = ActionChains(driver)
        driver.maximize_window()
        driver.execute_script("window.scrollTo(0, 50);")  # Scroll to the top

        start_element = driver.find_element(By.ID, "practice-last")
        actions.click(on_element=start_element).perform()
        actions.reset_actions()

        colors = [120, 161, 92, 131, 79, 71, 122]  # grayscale values of tetrominoes

        # Get intial observation
        obs = self.wrapped_env.env.reset()[0]

        sleep(2.5)
        while self.wrapped_env.env.game_over is False:
            # print(f"move {i}")
            driver.save_screenshot("image.png")

            box = (393,75,760,800) # around the tetris grid
            image = Image.open("image.png")
            image = image.crop(box)
            image = image.convert("L") # Convert to grayscale
            image.save("cropped_image.png")

            horizontal = box[2] - box[0]
            vertical = box[3] - box[1]
            image_pixels = image.load()
            
            # Get tetromino color and extract its index
            x = int(horizontal / 10 * 4) + 20
            y = int(vertical / 20 * 0) + 20
            tetromino_index = colors.index(image_pixels[x, y])
            
            # Replace the active tetromino in the wrapped environment
            state = self.wrapped_env.env.get_state()
            state.active_tetromino = self.wrapped_env.env.tetrominoes[tetromino_index]
            self.wrapped_env.env.set_state(state)

            """
            Passing observation to model and getting output
            """
            # Getting grouped observation from regular observation
            gobs = self.wrapped_env.observation(obs)

            # Converting observation to tensor and passing it through the model
            with torch.no_grad():
                obs_tensor = torch.tensor(gobs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 40, 13)
                q_values = self.policy(obs_tensor)  # (1, 40)
                action = q_values.argmax(dim=1).item()

            """
            Reenacting the move in environment game
            """
            self.wrapped_env.step(action)

            """
            Making the actual move in the online game
            """
            x, r = self.wrapped_env.decode_action(action)
            starting_x = 4 if tetromino_index == 1 else 3  # assume piece spawns at column 3 (check your online game)
            moves = x - starting_x

            # Rotate counterclockwise r times
            for _ in range(r):
                actions.send_keys("z")  # or whatever rotates the piece

            # Move left/right
            if moves < 0:
                for _ in range(abs(moves)):
                    actions.send_keys(Keys.ARROW_LEFT)
                    
            elif moves > 0:
                for _ in range(moves):
                    actions.send_keys(Keys.ARROW_RIGHT)

            # Hard drop
            actions.send_keys(Keys.SPACE)

            # Perform the actions
            actions.perform()
            actions.reset_actions()

        self.wrapped_env.env.render()
        cv2.waitKey(0)
        driver.quit()