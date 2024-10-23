# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Train agent with SARSA in Gym environment, making sequential decisions for maximizing cumulative rewards.

## SARSA LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with random values for all state-action pairs.

### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.

## SARSA LEARNING FUNCTION
```
DEVELOPED BY: Marella Dharanesh
REF NO: 212222240062
```
```python3
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state,action=next_state,next_action
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:
### optimal policy, optimal value function , success rate for the optimal policy.
![image](https://github.com/user-attachments/assets/60c041f2-bd4f-4d55-9dc1-4d261fe21c44)
![image](https://github.com/user-attachments/assets/72fedfde-17ec-4867-82da-a7dd19a51068)


### state value functions of Monte Carlo method:
![image](https://github.com/user-attachments/assets/7def5ddb-5749-4b24-8db1-83aaaf5eb17f)


### State value functions of SARSA learning:

![image](https://github.com/user-attachments/assets/8f34c971-44f8-40f4-94c8-1c7757cf6b7b)


## RESULT:
Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
