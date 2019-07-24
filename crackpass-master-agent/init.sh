#!/bin/sh

rabbitmqctl add_vhost vhost_crackpass
rabbitmqctl add_user crackpass crackpass
rabbitmqctl set_permissions -p vhost_crackpass crackpass ".*" ".*" ".*"
