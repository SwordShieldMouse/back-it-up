

def get_xyrange(envname):
    xmax = None

    if envname == 'ContinuousBandits':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Pendulum-v0':
        ymin = [-1600, -1600]
        ymax = [-100, -100]

    elif envname == 'Reacher-v2':
        ymin = [-110, -110]
        ymax = [0, 0]

    elif envname == "Swimmer-v2":
        ymin = [-5, -5]
        ymax = [35, 35]

    else:
        raise ValueError("Invalid environment name")

    if xmax is None:
        return None, ymin, ymax
    else:
        return xmax, ymin, ymax
