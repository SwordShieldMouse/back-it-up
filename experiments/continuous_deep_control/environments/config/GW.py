class GridWorldConfig():

    @staticmethod
    def config():
        return {"max_episode_len": 500,
                "num_rooms": 1,
                "action_limit_max": 1.0,
                "start_position": (8.0, 8.0),
                "goal_position": (22.0, 22.0),
                "goal_reward": +100.0,
                "dense_goals": [(13.0, 8.0), (18.0, 11.0), (20.0, 15.0), (22.0, 19.0)],
                "dense_reward": +5,
                "grid_len": 30}
