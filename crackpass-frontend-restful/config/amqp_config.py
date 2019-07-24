"""
Store database configuration
"""

import env_config


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


c_dev_config = {
   "auth": {
        "vhost": "vhost_crackpass",
        "username": "crackpass",
        "password": "crackpass"
    },
    "local": {
        "hostname": "localhost",
        "port": "5672"
    },
    "master":  {
        "hostname": "localhost",
        "port": "5672"
    }
}

c_test_config = {
   "auth": {
        "vhost": "vhost_crackpass",
        "username": "crackpass",
        "password": "crackpass"
    },
    "local": {
        "hostname": "localhost",
        "port": "5672"
    },
    "master":  {
        "hostname": "localhost",
        "port": "5672"
    }
}

c_product_config = {
   "auth": {
        "vhost": "vhost_crackpass",
        "username": "crackpass",
        "password": "crackpass"
    },
    "local": {
        "hostname": "localhost",
        "port": "5672"
    },
    "master":  {
        "hostname": "localhost",
        "port": "5672"
    }
}


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


env = env_config.get_conf()
selected_config = None
if env == "dev":
    selected_config = c_dev_config
elif env == "test":
    selected_config = c_test_config
elif env == "product":
    selected_config = c_product_config
else:
    print("Environment configuration error: no such environment ", env)
    exit()


def get_conf():
    return selected_config


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


if __name__ == '__main__':
    print(get_conf())
