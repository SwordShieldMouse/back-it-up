class GridWorldConfig():
    @staticmethod
    def config():
        return {"num_rooms": 1,
                "action_limit_max": 1.0,
                "start_position": (3.5, 3.5),
                "goal_position": (11.5, 3.5),
                "goal_reward": +10000.0,
                "dense_goals": [],
                "dense_reward": +50,
                "grid_len": 16}
