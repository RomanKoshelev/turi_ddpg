import gym
import tensorflow as tf
import config
from agent import AgentDDPG


def main():
  experiment = 'Pendulum-v0'
  env = gym.make(experiment)
  print(env.action_space)
  action_size = env.action_space.n
  print(env.observation_space)
  state_size = env.observation_space.n

  agent = AgentDDPG(env)

  for i in xrange(config.EPISODES):
    obs = env.reset()
    agent.set_state(obs)
    score = 0
    for t in xrange(config.STEPS):
      if config.SHOW_TRAINING:
        env.render()
        action = agent.get_action()
        # Execute action a_t and observe reward r_t and observe new observation s_{t+1}
        obs, reward, done, _ = env.step(action)
        score += reward

        # Store transition(s_t,a_t,r_t,s_{t+1}) and train the network
        agent.set_feedback(obs, action, reward, done)
        if done:
          print 'EPISODE: ', i, ' Steps: ', t, ' result: ', score
          score = 0
          break

if __name__ == '__main__':
  main()