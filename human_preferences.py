from torch import nn
import numpy as np
import datetime
import torch
import cv2
import gym
import os
from time import sleep
import pathlib
import matplotlib.pyplot as plt


class HumanPref(nn.Module):
    def __init__(self, obs_size, neuron_size=64):
        super(HumanPref, self).__init__()

        self.obs_size = obs_size
        self.neuron_size = neuron_size

        self.dense1 = nn.Linear(self.obs_size, self.neuron_size)
        self.dense2 = nn.Linear(self.neuron_size, 1)

        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x1, x2=None):

        model1_couche1 = self.dense1(x1)
        model1_couche2 = torch.nn.functional.relu(model1_couche1)
        model1_couche3 = self.dense2(model1_couche2)
        model1_couche4 = self.batch_norm(model1_couche3)
        if x2 is None:
            return model1_couche4
        else:
            model2_couche1 = self.dense1(x2)
            model2_couche2 = torch.nn.functional.relu(model2_couche1)
            model2_couche3 = self.dense2(model2_couche2)
            model2_couche4 = self.batch_norm(model2_couche3)
            # output = nn.functional.softmax(torch.stack([model1_couche4, model2_couche4]), dim=0)
            p1_sum = torch.exp(torch.sum(model1_couche1)/len(x1))
            p2_sum = torch.exp(torch.sum(model2_couche4)/len(x2))
            p1 = p1_sum/torch.add(p1_sum, p2_sum)
            p2 = p2_sum / torch.add(p1_sum, p2_sum)
            return torch.stack([p1, p2])


class HumanPreference(object):
    def __init__(self, obs_size, action_size):
        self.trijectories = []
        self.preferences = []
        self.layer_count = 3
        self.neuron_size_init = 64
        self.batch_size_init = 10
        self.learning_rate = 0.00025
        self.obs_size = obs_size
        self.action_size = action_size
        self.neuron_size = obs_size ** 3

        self.loss_l = []

        self.create_model()

    def create_model(self):
        self.model = HumanPref(self.obs_size, self.neuron_size)
        self.criterion = nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        if len(self.preferences) < 5:
            return

        batch_size = min(len(self.preferences), self.batch_size_init)
        r = np.asarray(range(len(self.preferences)))
        np.random.shuffle(r)

        min_loss = 1e+10
        max_loss = -1e+10
        lo = 0.0
        for i in r[:batch_size]:
            x0, x1, preference = self.preferences[i]

            pref_dist = np.zeros([2], dtype=np.float32)
            if preference < 2:
                pref_dist[preference] = 1.0
            else:
                pref_dist[:] = 0.5

            x0 = torch.from_numpy(np.asarray(x0)).float()
            x1 = torch.from_numpy(np.asarray(x1)).float()
            y = torch.from_numpy(pref_dist)
            y_hat = self.model(x0, x1)

            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() > max_loss:
                max_loss = loss.item()
            elif loss.item() < min_loss:
                min_loss = loss.item()

            lo = loss.item()
        print("[ Loss: actual loss =", lo, " max =", max_loss, " min =", min_loss, "]")

        self.loss_l.append(lo)

    def predict(self, obs):
        self.model.eval()
        obs = torch.tensor([obs]).float()
        pred = self.model(obs)
        return pred.detach().numpy()

    def add_preference(self, o0, o1, preference):
        self.preferences.append([o0, o1, preference])

    def add_trijactory(self, trijectory_env_name,  trijectory):
        self.trijectories.append([trijectory_env_name, trijectory])

    def ask_human(self):
        """
        Side-by-side trajectory playback in ONE OpenCV window (no Gym human windows).

        Overlay (per side):
        - elapsed time + step
        - cumulative reward
        - current action
        - DONE/RESET indicator

        Keys:
        1 -> prefer LEFT
        2 -> prefer RIGHT
        3 or 0 -> neutral
        SPACE -> pause/resume
        n -> single-step (only when paused)
        q or ESC -> quit without saving
        """
        import time

        if len(self.trijectories) < 2:
            return

        # pick two trajectories (random)
        r = np.arange(len(self.trijectories))
        np.random.shuffle(r)
        t = [self.trijectories[r[0]], self.trijectories[r[1]]]

        def unwrap_frame(frame):
            # gym/gymnasium may return a list of frames
            if isinstance(frame, list):
                return frame[-1] if len(frame) > 0 else None
            return frame

        def put_text(img, text, org, scale=0.7, color=(255, 255, 255), outline=(0, 0, 0), thickness=2):
            # Draw outline then main text (works on any background)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        envs = []
        try:
            # Create two envs in rgb_array mode (no Gym human windows)
            for i in range(2):
                env_name, _traj = t[i]
                env = gym.make(env_name, render_mode="rgb_array")
                _ = env.reset()  # gymnasium may return (obs, info)
                envs.append(env)

            win_name = "Preference (1=left,2=right,3=neutral | space=pause n=step | q/esc=quit)"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

            print("Preference (1,2|3). SPACE pause/resume, 'n' single-step, q/ESC quit (no save).")

            env_idxs = np.zeros([2], dtype=np.int32)
            preference = -1
            quit_flag = False

            # Overlay state
            start_time = time.time()
            paused = False
            step_once = False

            cum_rewards = [0.0, 0.0]
            last_action = [None, None]
            last_done_flag = [False, False]
            reset_flash = [0, 0]  # show DONE/RESET for N frames after reset/done

            delay_ms = 20

            while True:
                # ---- keyboard (always active) ----
                key = cv2.waitKey(delay_ms) & 0xFF

                if key == ord('1'):
                    preference = 0
                    break
                elif key == ord('2'):
                    preference = 1
                    break
                elif key == ord('3') or key == ord('0'):
                    preference = 2
                    break
                elif key == ord('q') or key == 27:  # ESC
                    quit_flag = True
                    break
                elif key == ord(' '):  # pause/resume
                    paused = not paused
                elif key == ord('n'):  # single-step (only when paused)
                    if paused:
                        step_once = True

                # ---- step logic ----
                do_step = (not paused) or step_once
                if do_step:
                    step_once = False

                    for i in range(2):
                        env_name, traj = t[i]

                        # traj element format expected: (obs, future_obs, action, done)
                        _obs, _future_obs, action, done_from_traj = traj[env_idxs[i]]
                        last_action[i] = action

                        # Step env (support both gym and gymnasium signatures)
                        step_out = envs[i].step(action)
                        if len(step_out) == 5:
                            _o, reward, terminated, truncated, _info = step_out
                            done_from_env = bool(terminated or truncated)
                        else:
                            _o, reward, done_from_env, _info = step_out
                            done_from_env = bool(done_from_env)

                        cum_rewards[i] += float(reward)
                        env_idxs[i] += 1

                        should_reset = (
                            bool(done_from_traj) or bool(done_from_env) or env_idxs[i] >= len(traj)
                        )
                        last_done_flag[i] = bool(done_from_traj) or bool(done_from_env)

                        if should_reset:
                            _ = envs[i].reset()
                            env_idxs[i] = 0
                            reset_flash[i] = 18  # show for ~18 frames
                            # If you want per-episode reward, uncomment:
                            # cum_rewards[i] = 0.0
                            last_done_flag[i] = True
                        else:
                            if reset_flash[i] > 0:
                                reset_flash[i] -= 1

                # ---- render frames ----
                frame0 = unwrap_frame(envs[0].render())
                frame1 = unwrap_frame(envs[1].render())

                if frame0 is None or frame1 is None:
                    both_bgr = np.zeros((240, 640, 3), dtype=np.uint8)
                    cv2.imshow(win_name, both_bgr)
                    continue

                # ---- resize to same height and stitch ----
                h = min(frame0.shape[0], frame1.shape[0])
                w0 = int(frame0.shape[1] * (h / frame0.shape[0]))
                w1 = int(frame1.shape[1] * (h / frame1.shape[0]))
                frame0r = cv2.resize(frame0, (w0, h), interpolation=cv2.INTER_AREA)
                frame1r = cv2.resize(frame1, (w1, h), interpolation=cv2.INTER_AREA)

                both_rgb = np.concatenate([frame0r, frame1r], axis=1)

                # ---- draw vertical separator line between two videos ----
                sep_x = frame0r.shape[1]
                cv2.line(both_rgb, (sep_x, 0), (sep_x, both_rgb.shape[0] - 1), (0, 0, 0), 3)          # black thick
                cv2.line(both_rgb, (sep_x, 0), (sep_x, both_rgb.shape[0] - 1), (255, 255, 255), 1)    # white thin center

                # ---- overlays ----
                elapsed = time.time() - start_time

                # Titles
                put_text(both_rgb, "LEFT", (10, 30), scale=1.0)
                put_text(both_rgb, "RIGHT", (sep_x + 10, 30), scale=1.0)

                # Pause indicator
                if paused:
                    put_text(both_rgb, "PAUSED (space resume, n step)", (10, 70), scale=0.8)

                # Per-side blocks (bottom)
                left_lines = [
                    f"time: {elapsed:.2f}s  step: {env_idxs[0]}",
                    f"reward_sum: {cum_rewards[0]:.3f}",
                    f"action: {last_action[0]}",
                ]
                right_lines = [
                    f"time: {elapsed:.2f}s  step: {env_idxs[1]}",
                    f"reward_sum: {cum_rewards[1]:.3f}",
                    f"action: {last_action[1]}",
                ]

                y_base = both_rgb.shape[0] - 70
                for k, line in enumerate(left_lines):
                    put_text(both_rgb, line, (10, y_base + 22 * k), scale=0.75)

                for k, line in enumerate(right_lines):
                    put_text(both_rgb, line, (sep_x + 10, y_base + 22 * k), scale=0.75)

                # DONE/RESET indicator (red text with outline) near top
                if last_done_flag[0] or reset_flash[0] > 0:
                    put_text(both_rgb, "DONE/RESET", (10, 110), scale=0.9, color=(255, 0, 0))
                if last_done_flag[1] or reset_flash[1] > 0:
                    put_text(both_rgb, "DONE/RESET", (sep_x + 10, 110), scale=0.9, color=(255, 0, 0))

                # Clear done flag after flash fades
                if reset_flash[0] == 0:
                    last_done_flag[0] = False
                if reset_flash[1] == 0:
                    last_done_flag[1] = False

                # ---- show ----
                both_bgr = both_rgb[:, :, ::-1]  # RGB -> BGR for OpenCV
                cv2.imshow(win_name, both_bgr)

            # ---- handle decision ----
            if quit_flag:
                print("quit (no preference saved)")
                return

            if preference != -1:
                # Keep your original behavior: store sequences of future_obs (index 1)
                os_list = []
                for i in range(2):
                    _env_name, traj = t[i]
                    o = [traj[j][1] for j in range(len(traj))]
                    os_list.append(o)
                self.add_preference(os_list[0], os_list[1], preference)

            if preference == 0:
                print("1 (left)")
            elif preference == 1:
                print("2 (right)")
            elif preference == 2:
                print("neutral")
            else:
                print("no opinion")

        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass




    def plot(self):
        x = np.arange(0, len(self.loss_l))
        y = np.asarray(self.loss_l)
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('Loss per epochs')

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(pathlib.Path().absolute(), 'plots', 'hp_model', 'hp_model' + datetime_str + ".png")
        plt.savefig(path)
