from gym.envs.registration import register

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v2',  # Environment name (including version number)
    entry_point='env_dag_train.fjsp_env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)
