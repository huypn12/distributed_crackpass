INSERT INTO auth_user( username, password, firstname, lastname, role, current_state, joined_DATE )
VALUES( 'anhth', SHA2('123', 256), 'tuan.anh', 'hoang', 'ADMIN', 'ACTIVE', NOW() );
