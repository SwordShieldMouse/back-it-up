

def get_xyrange(envname):
    xmax = None

    if envname == 'ContinuousBandits':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]
        yticks = None

    elif envname == 'Pendulum-v0':
        ymin = [-1600, -1600]
        ymax = [-100, -100]
        yticks = [a for a in range(-1500,-250+1,250)]

    elif envname == 'Reacher-v2':
        ymin = [-65, -65]
        ymax = [5, 5]
        yticks = [a for a in range(-60,-0+1,10)]

    elif envname == "Swimmer-v2":
        ymin = [0, 0]
        ymax = [40, 40]
        yticks = [a for a in range(10,40+1,10)]

    elif envname == "HalfCheetah-v2":
        ymin = [-1100, -1100]
        ymax = [3000, 3000]
        yticks = [a for a in range(-1000,3001,1000)]

    else:
        raise ValueError("Invalid environment name")

    if xmax is None:
        return None, ymin, ymax, yticks
    else:
        return xmax, ymin, ymax, yticks

def get_xyrange_with_p(envname, p):
    xmax = None

    if envname == 'ContinuousBandits':
        raise NotImplementedError

    elif envname == 'Pendulum-v0':
        if p == 'qf_vf_lr':
            ymin = [-1600, -1600]
            ymax = [-100, -100]
            yticks = [a for a in range(-1500,-250+1,250)]
        elif p == 'pi_lr':
            ymin = [-1300, -1300]
            ymax = [-100, -100]
            yticks = [a for a in range(-1250,-250+1,250)]            

    elif envname == 'Reacher-v2':
        if p == 'qf_vf_lr':
            ymin = [-125, -125]
            ymax = [5, 5]
            yticks = [a for a in range(-120,-0+1,30)]
        elif p == 'pi_lr':            
            ymin = [-40, -40]
            ymax = [5, 5]
            yticks = [a for a in range(-40,-0+1,10)]            

    elif envname == "Swimmer-v2":
        if p == 'qf_vf_lr':
            ymin = [-12, -12]
            ymax = [32, 32]
            yticks = [a for a in range(-10,30+1,10)]            
        elif p == 'pi_lr': 
            ymin = [7, 7]
            ymax = [32, 32]
            yticks = [a for a in range(10,30+1,10)]

    elif envname == "HalfCheetah-v2":
        if p == 'qf_vf_lr':
            ymin = [-1100, -1100]
            ymax = [2500, 2500]
            yticks = [a for a in range(-1000,2501,1000)]
        elif p == 'pi_lr': 
            ymin = [-1100, -1100]
            ymax = [1500, 1500]
            yticks = [a for a in range(-1000,1501,500)]

    else:
        raise ValueError("Invalid environment name")

    if xmax is None:
        return None, ymin, ymax, yticks
    else:
        return xmax, ymin, ymax, yticks