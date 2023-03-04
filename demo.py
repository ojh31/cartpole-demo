import glob
import gradio as gr
import gym
import sys
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch
from cartpole import (
    make_env, reset_env, Agent, rollout_phase, get_action_shape
)

MAIN = __name__ == "__main__"
examples = [0, 1, 31415, 'Hello, World!', 'This is a seed...']

def generate_video(
    string: str, wandb_path='wandb/run-20230303_211416-ox4d1p0u/files'
):
    with open(f'{wandb_path}/config.yaml') as f_cfg:
        config = yaml.safe_load(f_cfg)
    seed = hash(string)  % ((sys.maxsize + 1) * 2)
    num_envs = config['num_envs']['value']
    num_steps = config['num_steps']['value']
    assert seed >= 0
    assert isinstance(seed, int)
    run_name = f'seed{seed}'
    log_dir = f'generate/{run_name}'
    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([
        make_env("CartPole-v1", seed, i, True, run_name) 
        for i in range(num_envs)
    ])
    action_shape = get_action_shape(envs)
    next_obs, next_done = reset_env(envs, device)
    global_step = 0
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(f'{wandb_path}/model_state_dict.pt'))
    rollout_phase(
        next_obs, next_done, agent, envs, writer, device, 
        global_step, action_shape, num_envs, num_steps,
    )
    video_path = glob.glob(f'videos/{run_name}/*.mp4')[0]
    return video_path

if MAIN:
    demo = gr.Interface(
        fn=generate_video,
        inputs=[
            gr.components.Textbox(lines=1, label="Seed"),
        ],
        outputs=gr.components.Video(label="Generated Video"),
        examples=examples,
    )
    demo.launch(share=True)