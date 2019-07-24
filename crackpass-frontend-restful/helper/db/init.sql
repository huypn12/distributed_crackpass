# Created by @huypn
# This script is to initialize database for created schema

INSERT INTO auth_user( username, password, firstname, lastname, role, current_state, joined_date )
VALUES( 'huypn', SHA2('123', 256), 'huy', 'pn', 'ADMIN', 'ACTIVE', NOW() );

INSERT INTO auth_user( username, password, firstname, lastname, role, current_state, joined_DATE )
VALUES( 'ducnh', SHA2('123', 256), 'duc', 'nh', 'ADMIN', 'ACTIVE', NOW() );
