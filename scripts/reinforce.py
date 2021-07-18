from architecture.trainer import Trainer

LEARNING_RATES = [
  0.01,
  0.001,
  0.0001
]

for current in LEARNING_RATES:
  trainer = Trainer(learning_rate = current)
  trainer.train(num_episodes = 500)