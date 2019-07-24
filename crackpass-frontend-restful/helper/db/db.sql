# First created by @tuanhm
# Last edited by @huypn
# This script is to create schema for database.


#@huypn: replaced 'user' and 'status', as they are MySQL preserved words
#@huypn: add on update to last_login
## AUTHENTICATION MANIPULATING

DROP TABLE IF EXISTS auth_user;
DROP TABLE IF EXISTS auth_role;
DROP TABLE IF EXISTS user_has_role;
DROP TABLE IF EXISTS user_own_task;
DROP TABLE IF EXISTS task_has_type;
DROP TABLE IF EXISTS task;
DROP TABLE IF EXISTS task_type;
DROP TABLE IF EXISTS node;
DROP TABLE IF EXISTS task_on_node;


CREATE TABLE auth_user
(
    user_id         INT AUTO_INCREMENT,
    username        VARCHAR(100) UNIQUE,
    password        VARCHAR(256) NOT NULL,
    firstname       VARCHAR(50),
    lastname        VARCHAR(50),
    role            VARCHAR(20),
    current_state   VARCHAR(20),
    joined_date     DATETIME NOT NULL,
    last_login      TIMESTAMP NOT NULL  ON UPDATE CURRENT_TIMESTAMP,
    description     TEXT,

    # Constraints
    PRIMARY KEY (user_id)

) ENGINE=InnoDB;


## TASK MANIPULATION
CREATE TABLE task (
    # Fields
    task_id         INT AUTO_INCREMENT,
    type_name       VARCHAR(20) NOT NULL,
    n_cpu           INT,
    n_gpu           INT,
    params	        TEXT,
    result          TEXT,
    global_ctx         TEXT,
    progress        VARCHAR(128),
    eta             VARCHAR(128),
    current_state   VARCHAR(20),
    created_date    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    # Constraints
    PRIMARY KEY (task_id)
) ENGINE=InnoDB;


CREATE TABLE user_own_task (
    # Fields
    user_task_id INT AUTO_INCREMENT,
    user_id INT,
    task_id INT,

    # Constraints
    PRIMARY KEY (user_task_id),
    FOREIGN KEY (user_id) REFERENCES auth_user(user_id) ON UPDATE CASCADE,
    FOREIGN KEY (task_id) REFERENCES task(task_id) ON UPDATE CASCADE ON DELETE CASCADE

) ENGINE=InnoDB;


CREATE TABLE node (
    node_id INT AUTO_INCREMENT,
    node_name VARCHAR(64),
    ipaddr VARCHAR(50),
    amqp_url VARCHAR(512),
    total_gpu INT,
    total_cpu INT,
    avail_gpu INT,
    avail_cpu INT,

    PRIMARY KEY (node_id)
) ENGINE=InnoDB;


CREATE TABLE task_on_node(
    id INT AUTO_INCREMENT,
    task_id INT,
    node_id INT,
    n_cpu INT,
    n_gpu INT,
    local_ctx TEXT,

    PRIMARY KEY (id),
    FOREIGN KEY (task_id) REFERENCES task(task_id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (node_id) REFERENCES node(node_id) ON UPDATE CASCADE
) ENGINE=InnoDB;
