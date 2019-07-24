#!/usr/bin/python
import os

#@huypn: the following line is changed to test locally
#virtenv = os.environ['OPENSHIFT_PYTHON_DIR'] + '/virtenv/'
virtenv = os.path.join(os.environ.get('OPENSHIFT_PYTHON_DIR', '.'), 'virtenv')
virtualenv = os.path.join(virtenv, 'bin/activate_this.py')
try:
    execfile(virtualenv, dict(__file__=virtualenv))
except IOError:
    pass
#
# IMPORTANT: Put any additional includes below this line.  If placed above this
# line, it's possible required libraries won't be in your searchable path
#

from crackpass import app as application

#
# Below for testing only
#
if __name__ == '__main__':
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost', 8051, application)
    # Wait for a single request, serve it and quit.
    #@huypn: the following line is changed to serve request forever
    #httpd.handle_request()
    httpd.serve_forever()
