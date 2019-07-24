import requests, json

login_url = "http://localhost:5000/login"
data = json.dumps({
    'username': 'huypn',
    'password': 'xxx'
})
r = requests.post(login_url, data)
print("login result: ", r.json())

user_info_url = "http://localhost:5000/user/get_user_info"
data = json.dumps({
    'user_id': "2"
})

r = requests.post(user_info_url, data)
print("userinfo result: ", r)
