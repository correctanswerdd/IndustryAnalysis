from configparser import ConfigParser
import os

file_base = "../../10kdata"


def read_configuration(file_name='config.ini'):
    config = ConfigParser()
    config.read(file_name)

    parameters = {
        'extreme': config.getint('main', 'extreme'),
        'full': config.get('main', 'full'),
        'dir_logs': config.get('main', 'dir_logs'),
        'num_topics': [int(e) for e in config.get('main', 'num_topics').strip().split()],
        'layer_sizes': [int(e) for e in config.get('main', 'layer_sizes').strip().split()],
        'embedding_sizes': [int(e) for e in config.get('main', 'embedding_sizes').strip().split()],
        'batch_size': config.getint('main', 'batch_size'),
        'year': config.getint('main', 'year'),
        'num_epochs': config.getint('main', 'num_epochs'),
        'use_kl': config.getint('main', 'use_kl'),
    }

    return parameters


def save_configuration(parameters):
    # year = parameters["year"]
    # full = parameters["full"]
    # extreme = parameters['extreme']
    dir_logs = parameters['dir_logs']
    file_name = os.path.join(file_base, f"{dir_logs}/config.ini")

    config = ConfigParser()

    print(parameters)
    config.read(file_name)
    config.add_section('main')
    config.set('main', 'extreme', str(parameters["extreme"]))
    config.set('main', 'full', parameters["full"])
    config.set('main', 'dir_logs', parameters["dir_logs"])
    config.set('main', 'num_topics', parameters["num_topics"])
    config.set('main', 'layer_sizes', parameters["layer_sizes"])
    config.set('main', 'embedding_sizes', parameters["embedding_sizes"])
    config.set('main', 'batch_size', str(parameters["batch_size"]))
    config.set('main', 'year', str(parameters["year"]))
    config.set('main', 'num_epochs', str(parameters["num_epochs"]))
    config.set('main', 'use_kl', str(parameters["use_kl"]))

    with open(file_name, 'w') as f:
        config.write(f)
