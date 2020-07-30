from kaggle_environments import evaluate, make
env = make("halite", configuration={"episodeSteps": 400}, debug=True)
print(env.configuration)

print('test')
print('branch')
