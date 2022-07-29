import redis
import json


class MyRedis(redis.Redis):
    def set_dict(self, key: str, _dict: dict, **kwargs) -> bool:
        json_string = json.dumps(_dict, ensure_ascii=False).encode('utf-8')
        return self.set(key, json_string, **kwargs)

    def get_dict(self, key: str) -> dict:
        return json.loads(self.get(key))


if __name__ == '__main__':
    r = MyRedis(host='192.168.19.18', port=6379, db=0)
    r.set_dict('mydict', {'a': 'проверка', 'b_key': 17.2})
    mydict = r.get_dict('mydict')
    print(mydict)